# Klasifikasi Kematangan TBS dengan MobileNetV2

Training lokal, aplikasi Streamlit. Jalankan seluruh perintah dari folder proyek.

## 1. Instalasi

Gunakan Python 3.10-3.12 agar kompatibilitas TensorFlow lebih mudah.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
```

Windows PowerShell: aktivasi dengan `.venv\Scripts\Activate.ps1`.
GPU opsional; CPU bisa dipakai tetapi training lebih lambat. Instalasi TensorFlow standar tidak menjamin akselerasi GPU; cocokkan driver, sistem operasi dan petunjuk resmi TensorFlow.

## 2. Training lokal

```bash
python train.py
```

Script training mengunduh dataset versi 1 dengan kagglehub, lalu menyimpan hasil ke `hasil_tbs/` dan arsip `hasil_tbs.zip`. Internet dibutuhkan untuk unduhan awal dataset dan bobot ImageNet. Jika Kaggle meminta autentikasi, gunakan mekanisme resmi; jangan masukkan token ke source code.

Audit sampel/label sebelum eksperimen final. Dataset memiliki 857 citra, 6 duplikat piksel, 851 citra unik. Split dibangun ulang sekitar 70/15/15 setelah deduplikasi, bukan memakai CSV bawaan 70/20/10. Training script menyimpan grafik tanpa membuka jendela.

## 3. Aplikasi Streamlit

```bash
streamlit run app.py
```

Buka http://localhost:8501 atau alamat yang tercetak. Unggah citra lalu klik **Analisis kematangan**. Aplikasi menampilkan prediksi empat kelas, skor, status confidence threshold dan Grad-CAM. Model dimuat sekali menggunakan cache Streamlit.

Aplikasi membutuhkan tiga berkas di folder `hasil_tbs/`:

- `model.keras`
- `labels.json`
- `inference.json`

Jika model belum tersedia, aplikasi menampilkan petunjuk training, bukan gagal saat membaca file. Untuk menjalankan aplikasi saja, instal `requirements.txt`; tidak perlu dependensi training. Cocokkan versi TensorFlow/Keras dengan `hasil_tbs/environment.json`. Restart aplikasi setelah mengganti model agar cache diperbarui.

## Artefak hasil

- `audit.json`, `manifest_clean.csv`, `rejected.csv`: audit, ukuran dan distribusi data.
- `train.csv`, `validation.csv`, `test.csv`: split penelitian.
- `model.keras`, `labels.json`, `inference.json`: model terpilih, urutan kelas, threshold.
- `metrics.json`, `report_*.json`, `scenarios.csv`, `predictions.csv`: evaluasi nyata.
- `kurva.png`, `confusion_matrix.png`, `sampel.png`, `gradcam_*.png`: gambar laporan.
- `environment.json`, `architecture.txt`, `history_*.csv`: versi, arsitektur dan riwayat.

Label proyek: 0 mentah, 1 kurang matang, 2 matang, 3 lewat matang. Berbeda dari class_id sumber; pipeline memakai nama folder.

## Deployment publik: Streamlit Community Cloud

Lokal saja belum memenuhi syarat tautan aplikasi publik dalam format laporan.

1. Buat repository GitHub berisi app.py, requirements.txt dan tiga artefak model di hasil_tbs/. Aturan `.gitignore` menyertakan tiga berkas tersebut dan mengabaikan artefak training lainnya; tidak perlu `git add -f`. Jangan unggah kredensial.
2. Buka https://share.streamlit.io/, hubungkan repository, pilih app.py sebagai entrypoint. Samakan versi Python/dependensi dengan hasil training.
3. Tunggu build, uji prediksi benar/salah, status confidence rendah, dan Grad-CAM. Hosting gratis memiliki batas resource dan bukan jaminan layanan produksi.
4. Rekam tangkapan layar file model, build, UI dan hasil. Masukkan URL publik ke laporan; uji tanpa login.

## Pengujian

```bash
python -m unittest discover -s tests -v
```

Pengujian offline memeriksa sintaks, konsistensi inferensi aplikasi, sinkronisasi training, dan guard split. Bukan pengganti training atau evaluasi GPU.

## Sumber

Hadee Madadum, Oil Palm Fruit Ripeness Classification Dataset, versi 1, CC BY 4.0:
https://www.kaggle.com/datasets/hadeemadadum/oil-palm-fruit-ripeness-classification-dataset

MobileNetV2: https://arxiv.org/abs/1801.04381
Grad-CAM: https://arxiv.org/abs/1610.02391
Transfer learning: https://www.tensorflow.org/tutorials/images/transfer_learning
