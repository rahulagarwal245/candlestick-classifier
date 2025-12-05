import streamlit as st
import os
import numpy as np
from PIL import Image, ImageDraw
import json
import gdown
from tensorflow.keras.models import load_model

# ---------- CONFIG ----------
DRIVE_FILE_ID = "1cBgQDH-WbZuBEvBdoCwlgN3_YFiHxuHH"
LOCAL_MODEL = "models/best_candle_cnn.h5"
IMG_SIZE = (128, 128)
THRESHOLD = 0.55

st.set_page_config(page_title="Candlestick Classifier", layout="wide")

# ---------- HELPERS ----------
def download_model_if_missing():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(LOCAL_MODEL):
        url = "https://drive.google.com/uc?id=" + DRIVE_FILE_ID
        tmp = LOCAL_MODEL + ".part"
        gdown.download(url, tmp, quiet=False)
        os.replace(tmp, LOCAL_MODEL)

@st.cache_resource
def load_model_cached():
    download_model_if_missing()
    return load_model(LOCAL_MODEL)

def detect_last_candle(pil_img):
    img = pil_img.convert("RGB")
    w, h = img.size
    crop_w = max(10, int(w * 0.12))
    crop = img.crop((w - crop_w, 0, w, h))
    arr = np.array(crop)
    green_mask = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]) & (arr[:, :, 1] > 80)
    red_mask = (arr[:, :, 0] > arr[:, :, 1]) & (arr[:, :, 0] > arr[:, :, 2]) & (arr[:, :, 0] > 80)
    n_green = int(np.sum(green_mask))
    n_red = int(np.sum(red_mask))
    total = n_green + n_red
    if total == 0:
        return "Unknown", 0.0, n_green, n_red
    gf = n_green / total
    rf = n_red / total
    return ("Bullish", gf, n_green, n_red) if gf >= rf else ("Bearish", rf, n_green, n_red)

def interpret_conf(prob):
    if prob >= 0.85:
        return "Very High"
    if prob >= 0.65:
        return "High"
    if prob >= 0.45:
        return "Medium"
    return "Low"

def render_next_day_candle(pred_label, prob, width=120, height=220):
    """
    Generate a simple single-candle image (PIL) representing the next-day candle.
    - pred_label: "Bullish" or "Bearish"
    - prob: 0..1, used to scale body height
    """
    # Canvas
    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Visual metrics
    center_x = width // 2
    wick_top = 20
    wick_bottom = height - 20

    # Base candle body dimensions
    max_body_h = int((wick_bottom - wick_top) * 0.7)  # maximum body height
    min_body_h = int((wick_bottom - wick_top) * 0.08)  # minimum body height

    # Scale body height by prob (ensure some visibility for low prob)
    body_h = int(min_body_h + (max_body_h - min_body_h) * max(0.05, prob))
    body_w = int(width * 0.25)

    # Determine body top/bottom depending on bullish/bearish style:
    # For Bullish (green): open below close -> body bottom lower coordinate (bigger y)
    # For Bearish (red): open above close -> body top lower coordinate
    mid_y = (wick_top + wick_bottom) // 2
    body_top = mid_y - body_h // 2
    body_bottom = mid_y + body_h // 2

    # Colors
    green = (20, 120, 40)
    red = (180, 40, 40)
    wick_color = (60, 60, 60)

    body_color = green if pred_label == "Bullish" else red

    # Draw wick (line)
    draw.line([(center_x, wick_top), (center_x, body_top)], fill=wick_color, width=2)
    draw.line([(center_x, body_bottom), (center_x, wick_bottom)], fill=wick_color, width=2)

    # Draw body (rectangle)
    left = center_x - body_w // 2
    right = center_x + body_w // 2
    # Add small stroke for contrast
    draw.rectangle([left - 1, body_top - 1, right + 1, body_bottom + 1], outline=(0,0,0,30))
    draw.rectangle([left, body_top, right, body_bottom], fill=body_color)

    return img

# ---------- LOAD MODEL ----------
try:
    model = load_model_cached()
except Exception as e:
    st.error("Model loading failed: " + str(e))
    st.stop()

# ---------- PAGE HEADER ----------
st.markdown(
    "<h2 style='text-align:center; margin-bottom:6px;'>Candlestick Classifier</h2>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align:center; color:#666; margin-top:-6px;'>Upload a 30-candle chart to predict the next interval.</p>", unsafe_allow_html=True)

# ---------- UPLOADER ----------
uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded:
    pil = Image.open(uploaded).convert("RGB")

    # Larger display width for uploaded image:
    # On desktop this will show bigger; streamlit will stack columns on small screens.
    left_col, right_col = st.columns([2, 1], gap="large")

    # Left: larger chart
    with left_col:
        # Increase width to 720 on wide screens but cap for smaller images
        display_w = 720
        # keep aspect ratio
        display_h = int(display_w * pil.size[1] / pil.size[0])
        st.image(pil.resize((display_w, display_h)), caption="Uploaded Chart", use_column_width=False)

    # Right: current candle, prediction, next-day candle
    with right_col:
        cur_label, cur_conf, g_count, r_count = detect_last_candle(pil)
        st.markdown("### 🕯 Current Candle")
        st.write(f"Type: **{cur_label}**")
        st.write(f"Visual confidence: **{cur_conf:.2f}**")
        st.caption(f"Green: {g_count} | Red: {r_count}")

        # model prediction
        img128 = pil.resize(IMG_SIZE)
        arr = np.array(img128) / 255.0
        arr = np.expand_dims(arr, 0)
        prob = float(model.predict(arr)[0][0])
        pred = "Bullish" if prob >= THRESHOLD else "Bearish"
        prob_pct = prob * 100
        conf_text = interpret_conf(prob)

        st.markdown("---")
        st.markdown("### 📊 Next-day Prediction")
        color = "#2e7d32" if pred == "Bullish" else "#c62828"
        st.markdown(f"<div style='font-size:18px; font-weight:700; color:{color};'>{pred}</div>", unsafe_allow_html=True)
        st.write(f"Probability: **{prob_pct:.2f}%**")
        st.progress(min(max(prob, 0.0), 1.0))
        st.write(f"Confidence level: **{conf_text}**")

        # Render and show next-day candle
        next_candle_img = render_next_day_candle(pred, prob, width=140, height=260)
        st.markdown("#### Predicted Next-day Candle")
        st.image(next_candle_img, use_column_width=False)

        st.markdown("---")
        st.markdown("### 📝 Executive summary")
        summary = (
            f"The model predicts **{pred}** for the next interval with probability **{prob_pct:.1f}%**. "
            f"Confidence: **{conf_text}**. Use this as a short-term signal; not financial advice."
        )
        st.write(summary)

else:
    st.info("Upload a candlestick PNG/JPG to see prediction and the predicted next-day candle.")
