import streamlit as st
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown
import json

# ================= CONFIG =================
DRIVE_FILE_ID = "1cBgQDH-WbZuBEvBdoCwlgN3_YFiHxuHH"   # Your Google Drive model ID
LOCAL_MODEL = "models/best_candle_cnn.h5"
MODEL_STATS_PATH = "model_stats.json"
IMG_SIZE = (128, 128)
THRESHOLD = 0.55

st.set_page_config(page_title="Candlestick Classifier", layout="centered")
st.title("Candlestick Bullish / Bearish Classifier")

# ================= Utilities =================

def download_model_if_missing():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(LOCAL_MODEL):
        st.info("Model not found locally. Downloading from Google Drive (one-time)...")
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        tmp = LOCAL_MODEL + ".part"
        gdown.download(url, tmp, quiet=False)
        os.replace(tmp, LOCAL_MODEL)

@st.cache_resource
def load_model_cached():
    download_model_if_missing()
    model = load_model(LOCAL_MODEL)
    return model

def load_model_stats(path=MODEL_STATS_PATH):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return None
    return None

def detect_last_candle_from_image(pil_img):
    """
    Crop the rightmost region to detect green vs red pixels
    and classify the last candle visually.
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    crop_w = max(10, int(w * 0.12)*_
