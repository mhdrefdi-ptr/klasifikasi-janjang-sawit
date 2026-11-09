"""Run locally with: streamlit run app.py."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
SIZE = (224, 224)
@st.cache_resource
def load_artifacts(directory):
    model = keras.models.load_model(directory / 'model.keras', compile=False)
    labels = json.loads((directory / 'labels.json').read_text())
    threshold = json.loads((directory / 'inference.json').read_text())['threshold']
    if labels != ['mentah', 'kurang matang', 'matang', 'lewat matang'] or model.output_shape[-1] != 4:
        raise ValueError('Model dan labels.json tidak sesuai empat kelas proyek.')
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError('Threshold harus berupa angka antara 0 dan 1.')
    return model, labels, threshold


def gradcam(image):
    backbone = model.get_layer('mobilenetv2_1.00_224')
    probe = keras.Model(backbone.input, [backbone.get_layer('out_relu').output, backbone.output])
    x = tf.convert_to_tensor(np.asarray(image.resize(SIZE, Image.Resampling.BILINEAR).convert('RGB'), np.float32)[None])
    x = model.get_layer('mobilenet_scaling')(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        features, final = probe(x, training=False)
        pooled = model.get_layer('pool')(final)
        dense = model.get_layer('classifier')
        logits = tf.matmul(pooled, dense.kernel) + dense.bias
        index = tf.argmax(logits[0])
        score = logits[:, index]
    gradients = tape.gradient(score, features)
    if gradients is None:
        raise RuntimeError('Gradien Grad-CAM tidak tersambung.')
    heat = tf.reduce_sum(features[0] * tf.reduce_mean(gradients, axis=(0, 1, 2)), axis=-1)
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    heat = tf.image.resize(heat[..., None], SIZE).numpy().squeeze()
    colors = (plt.get_cmap('jet')(heat)[..., :3] * 255).astype(np.uint8)
    return Image.blend(image.convert('RGB').resize(SIZE, Image.Resampling.BILINEAR), Image.fromarray(colors), .4)


def predict_image(image):
    if image is None:
        raise ValueError('Unggah citra terlebih dahulu.')
    image = ImageOps.exif_transpose(image).convert('RGB')
    x = np.asarray(image.resize(SIZE, Image.Resampling.BILINEAR), np.float32)[None]
    probabilities = model.predict(x, verbose=0)[0]
    status = 'Perlu pemeriksaan manual' if probabilities.max() < threshold else ''
    return dict(zip(CLASSES, map(float, probabilities))), status, gradcam(image)


st.set_page_config(page_title='Kematangan TBS', layout='wide')
st.title('Klasifikasi Kematangan TBS')
st.caption('MobileNetV2 | Empat tingkat kematangan | Grad-CAM')
with st.sidebar:
    st.subheader('Tim PakaiAI')
    st.markdown(
        '1. Muhammad Refdi Pasaribu (231110267)\n'
        '2. Alberto William (231110810)\n'
        '3. Kevin Al Kautsar (231110461)\n'
        '4. Jethro Giovanno (231110940)'
    )
directory = ROOT / 'hasil_tbs'
required = ['model.keras', 'labels.json', 'inference.json']
if not all((directory / name).is_file() for name in required):
    st.info('Model belum tersedia. Jalankan python train.py. Hasil disimpan di folder hasil_tbs/.')
    st.stop()
try:
    model, CLASSES, threshold = load_artifacts(directory)
except (OSError, ValueError, TypeError, KeyError) as exc:
    st.error(f'Gagal memuat artefak: {exc}. Cocokkan versi TensorFlow/Keras dengan environment.json.')
    st.stop()
st.caption(f'Threshold pemeriksaan manual: {threshold:.2f}')
uploaded = st.file_uploader('Unggah foto tandan sawit', type=['jpg', 'jpeg', 'png', 'webp'])
if uploaded is not None:
    try:
        with Image.open(uploaded) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.load()
    except (OSError, ValueError, Image.DecompressionBombError):
        st.error('Berkas tidak dapat dibaca sebagai citra yang aman.')
        st.stop()
    input_column, result_column, gradcam_column = st.columns(3)
    preview = image.copy()
    preview.thumbnail((480, 360))
    with input_column:
        st.subheader('Citra masukan')
        st.image(preview)
        analyze = st.button('Analisis kematangan', type='primary')
    if analyze:
        with st.spinner('Menghitung prediksi dan Grad-CAM...'):
            probabilities, status, overlay = predict_image(image)
        label = max(probabilities, key=probabilities.get)
        with result_column:
            st.subheader('Hasil analisis')
            st.metric('Prediksi model', label.title())
            if probabilities[label] < threshold:
                st.warning(status)
            st.bar_chart(
                [{'Kelas': name, 'Confidence': score} for name, score in probabilities.items()],
                x='Kelas', y='Confidence', horizontal=True, height=240,
            )
        with gradcam_column:
            st.subheader('Grad-CAM')
            st.image(overlay, caption='Fokus visual model, bukan segmentasi atau bukti kausal.', width=320)
