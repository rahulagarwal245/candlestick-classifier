import streamlit as st
import os
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown
import json

# ===== CONFIG =====
DRIVE_FILE_ID = "1cBgQDH-WbZuBEvBdoCwlgN3_YFiHxuHH"
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
    Crop rightmost region of the image and count green vs red pixels
    to determine whether the last candle looks bullish or bearish.
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    crop_w = max(10, int(w * 0.12))
    crop = img.crop((w - crop_w, 0, w, h))
    arr = np.array(crop)

    # GREEN
    green_mask = (
        (arr[:, :, 1] > arr[:, :, 0]) &
        (arr[:, :, 1] > arr[:, :, 2]) &
        (arr[:, :, 1] > 80)
    )

    # RED
    red_mask = (
        (arr[:, :, 0] > arr[:, :, 1]) &
        (arr[:, :, 0] > arr[:, :, 2]) &
        (arr[:, :, 0] > 80)
    )

    n_green = int(np.count_nonzero(green_mask))
    n_red = int(np.count_nonzero(red_mask))
    total = max(1, n_green + n_red)

    # very low color presence → unknown
    if (n_green + n_red) < 0.01 * arr.shape[0] * arr.shape[1]:
        return "Unknown", 0.0, {"n_green": n_green, "n_red": n_red}

    green_frac = n_green / total
    red_frac = n_red / total

    if green_frac >= red_frac:
        return "Bullish", float(green_frac), {"n_green": n_green, "n_red": n_red}
    else:
        return "Bearish", float(red_frac), {"n_green": n_green, "n_red": n_red}


# ================= Load model + stats =================

try:
    model = load_model_cached()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

model_stats = load_model_stats()

# ================= UI =================

st.markdown("Upload a candlestick image (PNG/JPG). The model expects a 30-candle style chart image.")

uploaded = st.file_uploader("Upload candlestick image", type=["png", "jpg", "jpeg"])

if uploaded:

    pil_img = Image.open(uploaded).convert("RGB")
    w = 300
    h = int(w * pil_img.size[1] / pil_img.size[0])
    st.image(pil_img.resize((w, h)), caption="Uploaded Image", use_column_width=False)

    # ---- (1) Detect current candle visually ----
    current_label, current_conf, counts = detect_last_candle_from_image(pil_img)

    if current_label == "Unknown":
        st.write("**Current candle (visual):** Unable to determine.")
    else:
        st.write(f"**Current candle (visual):** {current_label}  (confidence: {current_conf:.2f})")
        st.caption(f"Detected pixels — Green: {counts['n_green']} | Red: {counts['n_red']}")

    # ---- (2) Model prediction ----
    img_resized = pil_img.resize(IMG_SIZE)
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, 0)

    prob = float(model.predict(x)[0][0])
    pred_label = "Bullish" if prob >= THRESHOLD else "Bearish"
    pred_pct = prob * 100.0

    st.write("---")
    st.write(f"**Next-day prediction:** {pred_label}")
    st.write(f"**Prediction probability:** {prob:.4f}  —  **{pred_pct:.1f}%**")

    # ---- (3) Show model metrics ----
    if model_stats:
        st.write("---")
        st.subheader("Model Test Metrics")
        if "test_accuracy" in model_stats:
            st.write(f"- Accuracy: **{model_stats['test_accuracy']:.3f}**")
        if "test_precision" in model_stats:
            st.write(f"- Precision: **{model_stats['test_precision']:.3f}**")
        if "test_recall" in model_stats:
            st.write(f"- Recall: **{model_stats['test_recall']:.3f}**")
        if "test_f1" in model_stats:
            st.write(f"- F1 Score: **{model_stats['test_f1']:.3f}**")
        if "num_test_samples" in model_stats:
            st.caption(f"Test samples used: {model_stats['num_test_samples']}")

    else:
        st.info("No model stats found — add model_stats.json to display accuracy and metrics.")
