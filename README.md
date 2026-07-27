# CIFAR-10 CNN Image Classifier

Aplikasi klasifikasi gambar berbasis web menggunakan Convolutional Neural Network (CNN) dengan Transfer Learning MobileNetV2, dilatih pada dataset CIFAR-10.

## Deskripsi

Proyek ini merupakan Tugas Kelompok 2 mata kuliah Artificial Intelligence (COSC6023036) — BINUS ONLINE 2026. Aplikasi menerima input gambar dari pengguna dan mengklasifikasikannya ke dalam 10 kelas: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, dan truck.

## Arsitektur Model

Model menggunakan Transfer Learning dengan MobileNetV2 (pretrained ImageNet) sebagai feature extractor, dilengkapi dengan Batch Normalization dan Dropout untuk regularisasi. Test accuracy yang dicapai adalah 86.04% pada dataset CIFAR-10.

## Teknologi

- Python
- TensorFlow / Keras
- Streamlit
- Google Colab (training)
- Streamlit Community Cloud (deployment)

## Struktur File

- `app.py` — Aplikasi web Streamlit
- `cnn_cifar10_best.keras` — Model CNN hasil training
- `requirements.txt` — Daftar dependensi Python
- `TK2_W8_S23_R0_Kelompok_2.ipynb` — Notebook training model

## Live Demo

https://tk2-group2.streamlit.app

## Cara Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```
