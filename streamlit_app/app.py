import streamlit as st
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
from pathlib import Path
import tempfile

# CONFIG
LOCAL_MODEL = "models/best_candle_cnn.h5"
MODEL_URL = ""  # if empty, expects LOCAL_MODEL present; otherwise put direct download URL
IMG_SIZE = (128,128)
THRESHOLD = 0.55

st.set_page_config(page_title="Candlestick Classifier", layout="centered")
st.title("Candlestick Bullish / Bearish Classifier")

@st.cache(allow_output_mutation=True)
def get_model(local_path=LOCAL_MODEL, url=MODEL_URL):
    # Ensure local folder
    Path("models").mkdir(parents=True, exist_ok=True)
    if not os.path.exists(local_path):
        if not url:
            st.error("Model file not found. Upload 'models/best_candle_cnn.h5' or set MODEL_URL.")
            return None
        # download model
        st.info("Downloading model (once). This may take a while...")
        r = requests.get(url, stream=True)
        tmp = local_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        os.rename(tmp, local_path)
    model = load_model(local_path)
    return model

model = get_model()
if model is None:
    st.stop()

uploaded = st.file_uploader("Upload candlestick image (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB").resize(IMG_SIZE)
    st.image(img, caption="Input image", use_column_width=False)
    x = np.array(img)/255.0
    x = np.expand_dims(x, 0)
    prob = float(model.predict(x)[0][0])
    label = "Bullish" if prob >= THRESHOLD else "Bearish"
    st.write(f"**Prediction:** {label}")
    st.write(f"**Probability:** {prob:.4f} (threshold={THRESHOLD})")
    if prob >= THRESHOLD:
        st.success("Model indicates bullish sentiment for the next interval.")
    else:
        st.error("Model indicates bearish sentiment for the next interval.")
