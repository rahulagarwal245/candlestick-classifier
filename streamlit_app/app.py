import streamlit as st
import os
import base64
import numpy as np
import json
from io import BytesIO
from PIL import Image
import gdown
from tensorflow.keras.models import load_model

# --------------- CONFIG ----------------
DRIVE_FILE_ID = "1cBgQDH-WbZuBEvBdoCwlgN3_YFiHxuHH"
LOCAL_MODEL = "models/best_candle_cnn.h5"
MODEL_STATS_PATH = "model_stats.json"
IMG_SIZE = (128, 128)
THRESHOLD = 0.55

st.set_page_config(page_title="Premium Candlestick Classifier", layout="centered")

# --------------- UTILITIES ----------------

def download_model_if_missing():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(LOCAL_MODEL):
        st.info("Downloading model...")
        url = "https://drive.google.com/uc?id=" + DRIVE_FILE_ID
        temp = LOCAL_MODEL + ".part"
        gdown.download(url, temp, quiet=False)
        os.replace(temp, LOCAL_MODEL)

@st.cache_resource
def load_model_cached():
    download_model_if_missing()
    return load_model(LOCAL_MODEL)

def load_model_stats():
    if not os.path.exists(MODEL_STATS_PATH):
        return None
    try:
        with open(MODEL_STATS_PATH, "r") as f:
            return json.load(f)
    except:
        return None

def detect_last_candle(pil_img):
    img = pil_img.convert("RGB")
    w, h = img.size
    crop_w = max(10, int(w * 0.12))
    crop = img.crop((w - crop_w, 0, w, h))
    arr = np.array(crop)

    gmask = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]) & (arr[:, :, 1] > 80)
    rmask = (arr[:, :, 0] > arr[:, :, 1]) & (arr[:, :, 0] > arr[:, :, 2]) & (arr[:, :, 0] > 80)

    n_green = int(np.sum(gmask))
    n_red = int(np.sum(rmask))
    total = n_green + n_red

    if total < 1:
        return "Unknown", 0.0, n_green, n_red

    gf = n_green / total
    rf = n_red / total
    return ("Bullish", gf, n_green, n_red) if gf >= rf else ("Bearish", rf, n_green, n_red)

def interpret_conf(prob):
    if prob >= 0.85:
        return "Very High Confidence"
    if prob >= 0.65:
        return "High Confidence"
    if prob >= 0.45:
        return "Medium Confidence"
    return "Low Confidence"

# --------------- LOAD MODEL ----------------

try:
    model = load_model_cached()
except Exception as e:
    st.error("Model loading failed: " + str(e))
    st.stop()

stats = load_model_stats()

# --------------- UI HEADER ----------------

st.markdown(
    "<h1 style='text-align:center; font-weight:700;'>Premium Candlestick Classifier</h1>",
    unsafe_allow_html=True
)

# --------------- FILE UPLOAD ----------------

file = st.file_uploader("Upload Candlestick Chart (PNG/JPG)", type=["png", "jpg", "jpeg"])

if file:
    pil_img = Image.open(file).convert("RGB")

    disp_w = 350
    disp_h = int(disp_w * pil_img.size[1] / pil_img.size[0])
    st.image(pil_img.resize((disp_w, disp_h)), caption="Uploaded Chart")

    # --- Current Candle ---
    label_now, conf_now, g_now, r_now = detect_last_candle(pil_img)

    st.subheader("🕯 Current Candle (Visual)")
    st.write("Type:", label_now)
    st.write("Confidence:", round(conf_now, 2))
    st.caption(f"Green: {g_now} | Red: {r_now}")

    # --- Prediction ---
    img128 = pil_img.resize(IMG_SIZE)
    arr = np.array(img128) / 255.0
    arr = np.expand_dims(arr, 0)

    prob = float(model.predict(arr)[0][0])
    pred = "Bullish" if prob >= THRESHOLD else "Bearish"
    prob_pct = prob * 100
    conf_text = interpret_conf(prob)

    # --- Executive Summary ---
    summary = (
        f"The model evaluated the uploaded candlestick pattern and "
        f"assigned a {pred} sentiment with a probability of {prob_pct:.1f}%. "
        f"The confidence is categorized as {conf_text}. "
        f"This interpretation is based on historically learned patterns."
    )

    # --- Glassmorphism Card ---
    card = """
    <div style='padding:20px; border-radius:15px;
    background: rgba(255,255,255,0.25);
    backdrop-filter: blur(12px);
    border: 2px solid;
    border-image: linear-gradient(135deg, #00E5FF, #005CFF) 1;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);'>
    """
    card += f"<h2 style='color:#005CFF;'>Prediction Summary</h2>"
    card += f"<p><b>Next-day Sentiment:</b> {pred}</p>"
    card += f"<p><b>Probability:</b> {prob_pct:.2f}%</p>"
    card += f"<p><b>Confidence:</b> {conf_text}</p>"
    card += f"<p><b>Model Accuracy:</b> {stats['test_accuracy']*100:.1f}%</p>"
    card += "</div>"

    st.markdown(card, unsafe_allow_html=True)

    # --- Model Metrics ---
    st.subheader("📈 Model Test Metrics")
    st.write(f"Accuracy: {stats['test_accuracy']:.3f}")
    st.write(f"Precision: {stats['test_precision']:.3f}")
    st.write(f"Recall: {stats['test_recall']:.3f}")
    st.write(f"F1 Score: {stats['test_f1']:.3f}")
    st.caption(f"Evaluated on {stats['num_test_samples']} samples.")

    # --- Executive Summary Display ---
    st.subheader("📝 Executive Summary")
    st.write(summary)
