"""Training lokal MobileNetV2. Jalankan: python train.py."""

from IPython.display import display

import os, json, hashlib, platform, importlib.metadata, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import kagglehub
SEED = 42
keras.utils.set_random_seed(SEED)
OUT = Path('hasil_tbs'); OUT.mkdir(exist_ok=True)
CLASSES = ['mentah', 'kurang matang', 'matang', 'lewat matang']
SIZE, BATCH = (224, 224), 32
print('TensorFlow:', tf.__version__, 'GPU:', tf.config.list_physical_devices('GPU'))
versions = {p: importlib.metadata.version(p) for p in ['tensorflow','keras','numpy','pandas','scikit-learn','pillow','streamlit','kagglehub']}
(OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(), 'packages':versions}, indent=2))
if __import__('shutil').which('nvidia-smi'):
    subprocess.run(['nvidia-smi'], check=False)

root = Path(kagglehub.dataset_download('hadeemadadum/oil-palm-fruit-ripeness-classification-dataset/versions/1'))
print('Folder:', sorted({p.parent.name for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}}))
LABEL_MAP = {'unripe':0, 'un-ripe':0, 'un ripe':0, 'mentah':0,
             'under-ripe':1, 'underripe':1, 'under ripe':1, 'kurang matang':1,
             'ripe':2, 'matang':2,
             'over-ripe':3, 'overripe':3, 'over ripe':3, 'lewat matang':3}
records, rejected = [], []
for p in sorted(root.rglob('*')):
    if p.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}: continue
    ancestors = [a.name.lower().replace('_',' ').strip() for a in p.parents if a != root.parent]
    labels = {LABEL_MAP[a] for a in ancestors if a in LABEL_MAP}
    if len(labels) != 1:
        raise ValueError(f'Label tidak dikenal/ambigu: {p}. Periksa LABEL_MAP.')
    try:
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB'); rgb.load()
            digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
            records.append({'path':str(p),'label':labels.pop(),'width':rgb.width,'height':rgb.height,'hash':digest})
    except (OSError, ValueError) as exc:
        rejected.append({'path':str(p),'error':str(exc)})
df = pd.DataFrame(records)
assert len(df) and set(df.label)==set(range(4)), 'Dataset harus memuat empat kelas.'
conflicts = df.groupby('hash').label.nunique()
assert not (conflicts>1).any(), 'Duplikat memiliki label berbeda; audit manual sebelum lanjut.'
n_raw = len(df)
df = df.drop_duplicates('hash').reset_index(drop=True)
assert df.groupby('label').size().min() >= 10, 'Data per kelas terlalu sedikit untuk split.'
df.to_csv(OUT/'manifest_clean.csv',index=False)
pd.DataFrame(rejected,columns=['path','error']).to_csv(OUT/'rejected.csv',index=False)
audit = {'valid_before_dedup':n_raw,'clean':len(df),'duplicates_removed':n_raw-len(df),'corrupt':len(rejected),
         'class_counts':{CLASSES[int(k)]:int(v) for k,v in df.label.value_counts().sort_index().items()},
         'dimensions':df[['width','height']].describe().to_dict()}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))

def savefig(name):
    plt.tight_layout(); plt.savefig(OUT/name,dpi=160,bbox_inches='tight'); plt.close('all')
df.label.value_counts().sort_index().rename(index=dict(enumerate(CLASSES))).plot.bar(title='Distribusi kelas setelah deduplikasi')
plt.ylabel('Jumlah citra'); savefig('distribusi.png')
fig, axes = plt.subplots(4,3,figsize=(10,12))
for label in range(4):
    for ax, row in zip(axes[label], df[df.label==label].sample(3,random_state=SEED).itertuples()):
        with Image.open(row.path) as im: ax.imshow(ImageOps.exif_transpose(im).convert('RGB'))
        ax.set_title(CLASSES[label]); ax.axis('off')
savefig('sampel.png')
df[['width','height']].hist(bins=20,figsize=(9,3)); savefig('resolusi.png')
train, hold = train_test_split(df,test_size=.30,stratify=df.label,random_state=SEED)
val, test = train_test_split(hold,test_size=.50,stratify=hold.label,random_state=SEED)
for a,b in [(train,val),(train,test),(val,test)]: assert not set(a.hash)&set(b.hash)
for name, frame in [('train',train),('validation',val),('test',test)]:
    frame.to_csv(OUT/f'{name}.csv',index=False)
print(pd.DataFrame({n:f.label.value_counts().sort_index() for n,f in [('train',train),('validation',val),('test',test)]}).rename(index=dict(enumerate(CLASSES))))

def read_rgb(path):
    with Image.open(path.decode()) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),dtype=np.float32)
def decode(path,label):
    image = tf.numpy_function(read_rgb,[path],tf.float32)
    image.set_shape((*SIZE,3))
    return image,label
def dataset(frame, training=False):
    ds=tf.data.Dataset.from_tensor_slices((frame.path.to_numpy(),frame.label.to_numpy(dtype=np.int32)))
    if training: ds=ds.shuffle(len(frame),seed=SEED)
    return ds.map(decode,num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)
train_ds, val_ds, test_ds = dataset(train,True), dataset(val), dataset(test)
weights=compute_class_weight('balanced',classes=np.arange(4),y=train.label)
class_weight=dict(enumerate(weights))

augmentation=keras.Sequential([
    keras.layers.RandomFlip('horizontal',seed=SEED),
    keras.layers.RandomRotation(.05,seed=SEED+1),
    keras.layers.RandomZoom(.10,seed=SEED+2)],name='augmentation')
base=keras.applications.MobileNetV2(input_shape=(*SIZE,3),include_top=False,weights='imagenet')
base.trainable=False
inputs=keras.Input((*SIZE,3),name='rgb_0_255')
x=augmentation(inputs)
x=keras.layers.Rescaling(1/127.5,offset=-1,name='mobilenet_scaling')(x)
x=base(x,training=False)
x=keras.layers.GlobalAveragePooling2D(name='pool')(x)
x=keras.layers.Dropout(.3)(x)
outputs=keras.layers.Dense(4,activation='softmax',name='classifier')(x)
model=keras.Model(inputs,outputs)
model.summary()
with open(OUT/'architecture.txt','w') as f: model.summary(print_fn=lambda s:f.write(s+'\n'))
def compile_model(rate):
    model.compile(optimizer=keras.optimizers.Adam(rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
def callbacks(name):
    return [keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=.5,min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(str(OUT/f'{name}.keras'),monitor='val_loss',save_best_only=True)]
compile_model(1e-3)
h1=model.fit(train_ds,validation_data=val_ds,epochs=15,class_weight=class_weight,callbacks=callbacks('frozen'),verbose=2)
frozen_val_loss=float(model.evaluate(val_ds,verbose=0)[0])
base.trainable=True
for layer in base.layers[:-30]: layer.trainable=False
for layer in base.layers:
    if isinstance(layer,keras.layers.BatchNormalization): layer.trainable=False
compile_model(1e-5)
h2=model.fit(train_ds,validation_data=val_ds,epochs=10,class_weight=class_weight,callbacks=callbacks('finetuned'),verbose=2)
finetuned_val_loss=float(model.evaluate(val_ds,verbose=0)[0])
selected='finetuned' if finetuned_val_loss < frozen_val_loss else 'frozen'
model=keras.models.load_model(OUT/f'{selected}.keras')
model.save(OUT/'model.keras')
print('Model dipilih berdasarkan validation loss:',selected)
(OUT/'labels.json').write_text(json.dumps(CLASSES))
for name,h in [('frozen',h1),('finetuned',h2)]:
    pd.DataFrame(h.history).to_csv(OUT/f'history_{name}.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,key in zip(axs,['loss','accuracy']):
    for prefix,h in [('train',key),('validation','val_'+key)]:
        values=h1.history[h]+h2.history[h]
        ax.plot(range(1,len(values)+1),values,label=prefix)
    ax.axvline(len(h1.history[key])+.5,color='gray',linestyle='--');ax.set_title(key);ax.set_xlabel('Epoch');ax.legend()
savefig('kurva.png')

vp=model.predict(val_ds,verbose=0); vy=val.label.to_numpy()
candidates=[]
for t in np.linspace(.25,.95,71):
    keep=vp.max(1)>=t
    if keep.mean()>=.5 and (vp.argmax(1)[keep]==vy[keep]).mean()>=.9:
        candidates.append(float(t))
threshold=min(candidates) if candidates else .5
(OUT/'inference.json').write_text(json.dumps({'threshold':threshold,'validation_target_met':bool(candidates),'selected':selected},indent=2))
print('Threshold:',threshold,'Target validation tercapai:',bool(candidates))

y=test.label.to_numpy()
scenario=[]
for name in ['frozen','finetuned']:
    candidate=keras.models.load_model(OUT/f'{name}.keras')
    p=candidate.predict(test_ds,verbose=0)
    report=classification_report(y,p.argmax(1),labels=range(4),target_names=CLASSES,output_dict=True,zero_division=0)
    scenario.append({'scenario':name,'accuracy':report['accuracy'],'macro_f1':report['macro avg']['f1-score']})
    (OUT/f'report_{name}.json').write_text(json.dumps(report,indent=2))
p=model.predict(test_ds,verbose=0); pred=p.argmax(1); keep=p.max(1)>=threshold
report=classification_report(y,pred,labels=range(4),target_names=CLASSES,output_dict=True,zero_division=0)
selective={'threshold':threshold,'accepted':int(keep.sum()),'total':len(y),'coverage':float(keep.mean()),
           'selective_accuracy':float((pred[keep]==y[keep]).mean()) if keep.any() else None,
           'accuracy_without_rejection':float((pred==y).mean())}
(OUT/'metrics.json').write_text(json.dumps({'selected_model':selected,'report':report,'selective':selective},indent=2))
pd.DataFrame(scenario).to_csv(OUT/'scenarios.csv',index=False)
print(pd.DataFrame(scenario)); print(pd.DataFrame(report).T); print(selective)
ConfusionMatrixDisplay(confusion_matrix(y,pred,labels=range(4)),display_labels=CLASSES).plot(xticks_rotation=30,cmap='Greens')
savefig('confusion_matrix.png')
results=test[['path','label']].copy(); results['prediction']=pred;results['confidence']=p.max(1);results['accepted']=keep
results.to_csv(OUT/'predictions.csv',index=False)
for correct in [True,False]:
    subset=results[(results.label==results.prediction)==correct].head(4)
    if subset.empty:
        print('Tidak ada contoh', 'benar' if correct else 'salah'); continue
    fig,axes=plt.subplots(1,len(subset),figsize=(4*len(subset),4),squeeze=False)
    for ax,row in zip(axes[0],subset.itertuples()):
        with Image.open(row.path) as im: ax.imshow(ImageOps.exif_transpose(im).convert('RGB'))
        ax.set_title(f'Asli: {CLASSES[row.label]}\nPrediksi: {CLASSES[row.prediction]} ({row.confidence:.2f})');ax.axis('off')
    savefig('prediksi_benar.png' if correct else 'prediksi_salah.png')

def gradcam(image):
    backbone=model.get_layer('mobilenetv2_1.00_224')
    probe=keras.Model(backbone.input,[backbone.get_layer('out_relu').output,backbone.output])
    x=tf.convert_to_tensor(np.asarray(image.resize(SIZE,Image.Resampling.BILINEAR).convert('RGB'),np.float32)[None])
    x=model.get_layer('mobilenet_scaling')(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        features, final=probe(x,training=False)
        pooled=model.get_layer('pool')(final)
        dense=model.get_layer('classifier')
        logits=tf.matmul(pooled,dense.kernel)+dense.bias
        index=tf.argmax(logits[0]);score=logits[:,index]
    gradients=tape.gradient(score,features)
    if gradients is None: raise RuntimeError('Gradien Grad-CAM tidak tersambung.')
    heat=tf.reduce_sum(features[0]*tf.reduce_mean(gradients,axis=(0,1,2)),axis=-1)
    heat=tf.maximum(heat,0);heat=heat/(tf.reduce_max(heat)+1e-8)
    heat=tf.image.resize(heat[...,None],SIZE).numpy().squeeze()
    colors=(plt.get_cmap('jet')(heat)[...,:3]*255).astype(np.uint8)
    return Image.blend(image.convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),Image.fromarray(colors),.4)
for correct in [True,False]:
    subset=results[(results.label==results.prediction)==correct].head(1)
    for row in subset.itertuples():
        with Image.open(row.path) as im: overlay=gradcam(ImageOps.exif_transpose(im).convert('RGB'))
        overlay.save(OUT/('gradcam_benar.png' if correct else 'gradcam_salah.png'))
        display(overlay)

import shutil
shutil.make_archive('hasil_tbs','zip',OUT)
print('Hasil tersimpan di hasil_tbs/ dan hasil_tbs.zip')
