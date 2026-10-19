import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import tensorflow as tf

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️", layout="wide")
MODEL_PATH = Path(__file__).parent / "handwrite_cnn_model.h5"
DIGIT_LABELS = [str(i) for i in range(10)]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#172554,#1d4ed8); color: white; margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:.5rem 0 0 0; opacity:.9;}
</style>
<div class="hero"><h1>✍️ Handwritten Digit Recognition</h1><p>Upload an image of a handwritten digit and let your trained CNN predict the digit from 0 to 9.</p></div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image(image: Image.Image):
    # Notebook training uses RGB images, 128×128 size, and pixel rescaling by 1/255.
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.fit(image, (128, 128), method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0), image

with st.sidebar:
    st.header("About the model")
    st.write("**Architecture:** Convolutional Neural Network (CNN)")
    st.write("**Input:** 128 × 128 RGB image")
    st.write("**Output:** Digit classes 0–9")
    st.write("**Preprocessing:** Pixel values scaled to 0–1")
    st.caption("Prediction quality depends on the image resembling the training data.")

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("1. Upload an image")
    uploaded = st.file_uploader("Choose a PNG, JPG, JPEG, or WEBP image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        try:
            original = Image.open(uploaded)
            st.image(original, caption="Uploaded image", use_container_width=True)
        except Exception:
            st.error("This file could not be opened as an image.")
            uploaded = None
    else:
        st.info("For best results, upload a clear image containing one handwritten digit.")

    st.subheader("2. Predict the digit")
    if uploaded:
        if st.button("🔍 Predict Digit", type="primary", use_container_width=True):
            try:
                model = load_model()
                tensor, processed = preprocess_image(Image.open(uploaded))
                output = np.asarray(model.predict(tensor, verbose=0))[0].reshape(-1)
                if len(output) != 10:
                    st.error(f"The model returned {len(output)} outputs; this dashboard expects 10 digit classes.")
                else:
                    idx = int(np.argmax(output))
                    confidence = float(output[idx])
                    st.success("Prediction complete")
                    st.metric("Predicted digit", DIGIT_LABELS[idx], f"Confidence: {confidence:.1%}")
                    st.image(processed, caption="Image after resize and RGB conversion", width=220)
                    st.markdown("**Class scores**")
                    scores = {f"Digit {i}": float(output[i]) for i in range(10)}
                    st.bar_chart(scores)
                    st.caption("Scores are the model's softmax outputs; they are not a guarantee of correctness.")
            except Exception as exc:
                st.error("Could not load the model or run prediction.")
                st.exception(exc)
    else:
        st.button("🔍 Predict Digit", type="primary", disabled=True, use_container_width=True)

with st.expander("How to use this dashboard"):
    st.markdown("1. Upload an image with one handwritten digit.\n2. Click **Predict Digit**.\n3. Review the predicted digit and the score distribution.\n\nThe dashboard uses the same input size (128×128) and normalization (divide pixels by 255) shown in your notebook.")
