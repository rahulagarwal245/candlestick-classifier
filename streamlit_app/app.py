# streamlit_app/app.py
import streamlit as st
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown
import json

# ===== CONFIG =====
DRIVE_FILE_ID = "1cBgQDH-WbZuBEvBdoCwlgN3_YFiHxuHH"  # your Drive id (already set)
LOCAL_MODEL = "models/best_candle_cnn.h5"
MODEL_STATS_PATH = "model_stats.json"   # file you created in repo root
IMG_SIZE = (128, 128)
THRESHOLD = 0.55

st.set_page_config(page_title="Candlestick Classifier", layout="centered")
st.title("Candlestick Bullish / Bearish Classifier")

# ---------- Utilities ----------
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
        except Exception:
            return None
    return None

def detect_last_candle_from_image(pil_img):
    """
    Heuristic: crop rightmost region and count green vs red pixels.
    Returns label (Bullish/Bearish/Unknown), confidence, and counts.
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    crop_w = max(10, int(w * 0.12))
    crop = img.crop((w - crop_w, 0, w, h))
    arr = np.array(crop)

    green_mask = ((arr[:,:,1] > arr[:,:,0]) & (arr[:,:,1] > arr[:,:,2]) & (arr[:,:,1] > 80))
    red_mask = ((arr[:,:,0] > arr[:,:,1]) & (arr[:,:,0] > arr[:,:,2]) & (arr[:,:,0] > 80))

    n_green = int(np.count_nonzero(green_mask))
    n_red = int(np.count_nonzero(red_mask))
    total = max(1, n_green + n_red)
    green_frac = n_green / total
    red_frac = n_red / total

    if (n_green + n_red) < (0.01 * arr.shape[0] * arr.shape[1]):
        return "Unknown", 0.0, {"n_green": n_green, "n_red": n_red, "total_color": n_green + n_red}

    if green_frac >= red_frac:
        return "Bullish", float(green_fr_
