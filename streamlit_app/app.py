import streamlit as st
import os
import base64
import numpy as np
import json
from io import BytesIO
from PIL import Image
from fpdf import FPDF
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

def pil_to_bytes(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# --------------- PDF GENERATOR (R3 Minimal Black & White) ----------------

def generate_pdf(pred, prob, conf_text, accuracy, summary, pil_img):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Candlestick Prediction Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 8, f"Prediction: {pred}", ln=True)
    pdf.cell(0, 8, f"Probability: {prob:.2f}%", ln=True)
    pdf.cell(0, 8, f"Confidence Level: {conf_text}", ln=True)
    pdf.cell(0, 8, f"Model Accuracy: {accuracy:.1f}%", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Executive Summary", ln=True)

    pdf.set_font("Arial", "", 12)
    for line in summary.split("\n"):
        pdf.multi_cell(0, 7, line)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Uploaded Candle Image", ln=True)

    img_bytes = pil_to_bytes(pil_img)
    img_path = "temp_img.png"
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    pdf.image(img_path, w=120)

    os.remove(img_path)

    return pdf.output(dest="S").encode("latin-1")

# --------------- LOAD MODEL + STATS ----------------

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

    # --- Premium Glassmorphism Card ---
    card_html = """
    <div style='padding:20px; border-radius:15px; 
    background: rgba(255,255,255,0.25); 
    backdrop-filter: blur(12px);
    border: 2px solid; 
    border-image: linear-gradient(135deg, #00E5FF, #005CFF) 1;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);'>
    """

    card_html += f"<h2 style='color:#005CFF;'>Prediction Summary</h2>"
    card_html += f"<p><b>Next-day Sentiment:</b> {pred}</p>"
    card_html += f"<p><b>Probability:</b> {prob_pct:.2f}%</p>"
    card_html += f"<p><b>Confidence:</b> {conf_text}</p>"
    card_html += f"<p><b>Model Accuracy:</b> {stats['test_accuracy']*100:.1f}%</p>"
    card_html += "</div>"

    st.markdown(card_html, unsafe_allow_html=True)

    # --- Metrics ---
    st.subheader("📈 Model Test Metrics")
    st.write(f"Accuracy: {stats['test_accuracy']:.3f}")
    st.write(f"Precision: {stats['test_precision']:.3f}")
    st.write(f"Recall: {stats['test_recall']:.3f}")
    st.write(f"F1 Score: {stats['test_f1']:.3f}")
    st.caption(f"Evaluated on {stats['num_test_samples']} samples.")

    # --- PDF Download ---
    st.subheader("📥 Download PDF Report")

    pdf_bytes = generate_pdf(
        pred=pred,
        prob=prob_pct,
        conf_text=conf_text,
        accuracy=stats["test_accuracy"]*100,
        summary=summary,
        pil_img=pil_img
    )

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="candlestick_report.pdf",
        mime="application/pdf"
    )
