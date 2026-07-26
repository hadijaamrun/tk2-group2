import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# =========================================================
# KONFIGURASI
# =========================================================
MODEL_PATH = "cnn_cifar10_best.keras"   
IMG_SIZE = (32, 32)                     
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.set_page_config(page_title="CIFAR-10 CNN Classifier", page_icon="🖼️", layout="centered")


# =========================================================
# LOAD MODEL 
# =========================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocessing HARUS sama persis dengan preprocessing saat training:
    - resize ke 32x32
    - convert ke RGB (jaga-jaga kalau user upload PNG dengan alpha channel)
    - normalisasi ke [0, 1]
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0) 
    return arr


# =========================================================
# UI
# =========================================================
st.title("🖼️ CIFAR-10 CNN Image Classifier")
st.write(
    "Upload sebuah gambar (pesawat, mobil, burung, kucing, rusa, anjing, "
    "katak, kuda, kapal, atau truk), lalu model CNN akan memprediksi kelasnya."
)

with st.spinner("Memuat model..."):
    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"Gagal memuat model dari '{MODEL_PATH}'. Pastikan file model ada di folder yang sama. Detail error: {e}")

uploaded_file = st.file_uploader("Upload gambar di sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

    with st.spinner("Memproses prediksi..."):
        input_arr = preprocess_image(image)
        predictions = model.predict(input_arr, verbose=0)[0]

    pred_idx = int(np.argmax(predictions))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(predictions[pred_idx]) * 100

    with col2:
        st.subheader("Hasil Prediksi")
        st.success(f"**{pred_class.upper()}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        # Tampilkan top-3 prediksi
        st.write("**Top-3 kemungkinan:**")
        top3_idx = np.argsort(predictions)[::-1][:3]
        for idx in top3_idx:
            st.write(f"- {CLASS_NAMES[idx]}: {predictions[idx]*100:.2f}%")

    # Bar chart semua kelas
    st.write("**Distribusi probabilitas semua kelas:**")
    st.bar_chart({CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))})

elif uploaded_file is not None and not model_loaded:
    st.warning("Model belum berhasil dimuat, prediksi tidak bisa dijalankan.")
else:
    st.info("Silakan upload gambar untuk memulai prediksi.")

st.markdown("---")
st.caption("Model CNN dilatih pada dataset CIFAR-10 (32x32 piksel, 10 kelas).")
