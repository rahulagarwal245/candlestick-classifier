# streamlit_app/app.py
import streamlit as st
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown

# ===== CONFIG =====
DRIVE_FILE_ID = "1cBgQDH-WbZuBEvBdoCwlgN3_YFiHxuHH"  # <<-- your Google Drive file id
LOCAL_MODEL = "models/best_candle_cnn.h5"
IMG_SIZE = (128, 128)
THRESHOLD = 0.55

st.set_page_config(page_title="Candlestick Classifier", layout="centered")
st.title("Candlestick Bullish / Bearish Classifier")

@st.cache(allow_output_mutation=True)
def load_or_download_model():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(LOCAL_MODEL):
        st.info("Model not found locally. Downloading model from Google Drive (one-time)...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        tmp_path = LOCAL_MODEL + ".part"
        # download via gdown; this handles large files robustly
        gdown.download(url, tmp_path, quiet=False)
        os.replace(tmp_path, LOCAL_MODEL)

    # load the Keras model
    model = load_model(LOCAL_MODEL)
    return model

# Load model (download if required)
try:
    model = load_or_download_model()
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

# ===== UI =====
st.markdown("Upload a candlestick image (PNG/JPG). The model expects a 30-candle style chart image.")
uploaded = st.file_uploader("Upload candlestick image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB").resize(IMG_SIZE)
    st.image(img, caption="Uploaded Image", use_column_width=False)

    x = np.array(img) / 255.0
    x = np.expand_dims(x, 0)

    prob = float(model.predict(x)[0][0])
    label = "Bullish" if prob >= THRESHOLD else "Bearish"

    st.write(f"**Prediction:** {label}")
    st.write(f"**Probability:** {prob:.4f} (Threshold = {THRESHOLD})")

    if prob >= THRESHOLD:
        st.success("Model indicates bullish sentiment for the next interval.")
    else:
        st.error("Model indicates bearish sentiment for the next interval.")
