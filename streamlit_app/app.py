import streamlit as st
import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import pdfkit
import gdown
import json
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
DRIVE_FILE_ID = "1cBgQDH-WbZuBEvBdoCwlgN3_YFiHxuHH"
LOCAL_MODEL = "models/best_candle_cnn.h5"
MODEL_STATS_PATH = "model_stats.json"
IMG_SIZE = (128, 128)
THRESHOLD = 0.55

st.set_page_config(page_title="Premium Candlestick Classifier", layout="wide")

# ---------------- UTILITIES ----------------

def download_model_if_missing():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(LOCAL_MODEL):
        st.info("Downloading model (one-time)...")
        url = "https://drive.google.com/uc?id=" + DRIVE_FILE_ID
        temp = LOCAL_MODEL + ".part"
        gdown.download(url, temp, quiet=False)
        os.replace(temp, LOCAL_MODEL)

@st.cache_resource
def load_model_cached():
    download_model_if_missing()
    return load_model(LOCAL_MODEL)

def load_model_stats():
    if os.path.exists(MODEL_STATS_PATH):
        try:
            with open(MODEL_STATS_PATH, "r") as f:
                return json.load(f)
        except:
            return None
    return None

def detect_last_candle(pil_img):
    img = pil_img.convert("RGB")
    w, h = img.size
    crop_w = int(w * 0.12)
    if crop_w < 10:
        crop_w = 10
    crop = img.crop((w - crop_w, 0, w, h))
    arr = np.array(crop)

    green_mask = (arr[:, :, 1] > arr[:, :, 0]) & (arr[:, :, 1] > arr[:, :, 2]) & (arr[:, :, 1] > 80)
    red_mask = (arr[:, :, 0] > arr[:, :, 1]) & (arr[:, :, 0] > arr[:, :, 2]) & (arr[:, :, 0] > 80)

    n_green = int(np.sum(green_mask))
    n_red = int(np.sum(red_mask))
    total = n_green + n_red

    if total < 1:
        return "Unknown", 0.0, n_green, n_red

    gf = n_green / total
    rf = n_red / total

    if gf >= rf:
        return "Bullish", gf, n_green, n_red
    return "Bearish", rf, n_green, n_red

def interpret_confidence(prob):
    if prob >= 0.85:
        return "Very High Confidence"
    if prob >= 0.65:
        return "High Confidence"
    if prob >= 0.45:
        return "Medium Confidence"
    return "Low Confidence"

def image_to_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()


# ---------------- LOAD MODEL ----------------
try:
    model = load_model_cached()
except Exception as e:
    st.error("Model load error: " + str(e))
    st.stop()

stats = load_model_stats()


# ---------------- UI HEADER ----------------
st.markdown(
    """
    <h1 style='font-weight:700; text-align:center;'>
        Premium Candlestick Classifier (Glassmorphism)
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Upload a candlestick image to generate prediction and a downloadable PDF report.")


# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload candlestick chart (PNG/JPG)", type=["png", "jpg", "jpeg"])

if file:

    # Load & display
    pil_img = Image.open(file).convert("RGB")
    disp_w = 350
    disp_h = int(disp_w * pil_img.size[1] / pil_img.size[0])
    st.image(pil_img.resize((disp_w, disp_h)), caption="Uploaded Chart")

    # -------- CURRENT CANDLE --------
    label_now, conf_now, g_now, r_now = detect_last_candle(pil_img)

    st.subheader("🕯 Current Candle (Visual)")
    st.write("Type:", label_now)
    st.write("Confidence:", round(conf_now, 2))
    st.caption(f"Green pixels: {g_now} | Red pixels: {r_now}")

    # -------- MODEL PREDICTION --------
    img128 = pil_img.resize(IMG_SIZE)
    x = np.array(img128) / 255.0
    x = np.expand_dims(x, 0)

    prob = float(model.predict(x)[0][0])
    pred = "Bullish" if prob >= THRESHOLD else "Bearish"
    pred_pct = prob * 100
    conf_text = interpret_confidence(prob)

    # -------- EXECUTIVE SUMMARY --------
    exec_summary = f"""
    The model identifies the uploaded candlestick structure and market sentiment.
    Based on learned pattern behaviors, the next-day trend is assessed as **{pred}** with a probability of **{pred_pct:.1f}%**.
    Confidence level is categorized as **{conf_text}**, reflecting how strongly the model perceives pattern alignment with historical bullish/bearish signals.
    """

    # -------- GLASS CARD --------
    glass_card = f"""
    <div style="
        padding:20px;
        border-radius:18px;
        backdrop-filter: blur(12px);
        background: rgba(255,255,255,0.28);
        border: 2px solid;
        border-image: linear-gradient(135deg, #00E5FF, #005CFF) 1;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        margin-top:20px;">
        <h2 style="color:#005CFF;">Prediction Summary</h2>
        <p><b>Next-day sentiment:</b> {pred}</p>
        <p><b>Probability:</b> {pred_pct:.2f}%</p>
        <p><b>Confidence Level:</b> {conf_text}</p>
        <p><b>Model Accuracy:</b> {stats["test_accuracy"]*100:.1f}%</p>
    </div>
    """

    st.markdown(glass_card, unsafe_allow_html=True)

    # -------- METRICS GRID --------
    st.subheader("📈 Model Test Metrics")
    st.write(f"Accuracy: {stats['test_accuracy']:.3f}")
    st.write(f"Precision: {stats['test_precision']:.3f}")
    st.write(f"Recall: {stats['test_recall']:.3f}")
    st.write(f"F1 Score: {stats['test_f1']:.3f}")
    st.caption(f"Evaluated on {stats['num_test_samples']} samples.")

    # -------- PDF GENERATION --------
    st.subheader("📥 Download PDF Report")

    base64_img = image_to_base64(pil_img)

    html_report = f"""
    <html>
    <body>
        <h1>Premium Candlestick Report</h1>
        <h2>Prediction Summary</h2>
        <p><b>Prediction:</b> {pred}</p>
        <p><b>Probability:</b> {pred_pct:.2f}%</p>
        <p><b>Confidence Level:</b> {conf_text}</p>
        <p><b>Model Accuracy:</b> {stats["test_accuracy"]*100:.1f}%</p>

        <h2>Uploaded Chart</h2>
        <img src="data:image/png;base64,{base64_img}" width="350">

        <h2>Executive Summary</h2>
        <p>{exec_summary}</p>
    </body>
    </html>
    """

    pdf_bytes = pdfkit.from_string(html_report, False)

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="candlestick_report.pdf",
        mime="application/pdf"
    )
