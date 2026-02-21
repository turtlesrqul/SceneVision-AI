"""
Intel Image Classification - Streamlit Web App
================================================
Deploys the best-performing Keras model as an interactive image classifier.
Upload a scene photograph and the model predicts one of six categories:
buildings, forest, glacier, mountain, sea, or street.

Run with:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf

# ── Configuration ────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/best_model.keras")
IMG_SIZE = (150, 150)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_ICONS = {
    "buildings": "🏢", "forest": "🌲", "glacier": "🏔️",
    "mountain": "⛰️", "sea": "🌊", "street": "🛣️",
}


# ── Load model (cached so it only loads once) ────────────────
@st.cache_resource
def load_model():
    """Load the trained Keras model from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at `{MODEL_PATH}`. "
                 "Check that the path is correct and the file exists.")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess an uploaded PIL image to match training pipeline:
      1. Convert to RGB (handles RGBA, grayscale, etc.)
      2. Resize to 150x150
      3. Convert to float32 numpy array
      4. Scale pixel values to [0, 1]
      5. Add batch dimension -> shape (1, 150, 150, 3)
    """
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ── Streamlit UI ─────────────────────────────────────────────
st.set_page_config(
    page_title="Intel Image Classifier",
    page_icon="🖼️",
    layout="centered",
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-card h2 {
        font-size: 2rem;
        margin: 0;
        color: white;
    }
    .prediction-card p {
        font-size: 1.1rem;
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
    }
    .prob-bar-container {
        margin: 0.4rem 0;
    }
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        margin-bottom: 2px;
    }
    .prob-bar {
        background: #e5e7eb;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 1rem;
    }
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About This App")
    st.markdown(
        "SceneVision is an AI-powered scene recognition tool built for "
        "geospatial analysts, urban planners, and environmental researchers "
        "who need to quickly categorise large volumes of landscape imagery."
    )
    st.markdown("---")
    st.markdown("#### How It Works")
    st.markdown(
        "1. **Upload** a photograph of any natural or urban scene.\n"
        "2. **Our model** analyses the image in under a second.\n"
        "3. **Get results** with confidence scores for all six categories."
    )
    st.markdown("---")
    st.markdown("#### Model Details")
    st.markdown(
        "| Spec | Value |\n"
        "|---|---|\n"
        "| Architecture | EfficientNetB0 (fine-tuned) |\n"
        "| Input Size | 150 x 150 px |\n"
        "| Classes | 6 |\n"
        "| Training Images | ~14,000 |\n"
        "| Framework | TensorFlow / Keras |\n"
    )
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#9ca3af;font-size:0.8rem;'>"
        "Built with Streamlit<br>Powered by TensorFlow"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Main content ─────────────────────────────────────────────
# Header
st.markdown('<div class="main-header"><h1>🖼️ SceneVision AI</h1></div>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Instant scene recognition powered by deep learning. '
    'Upload any landscape or urban photograph and our model will identify it in seconds.</p>',
    unsafe_allow_html=True,
)

# Category chips
cat_display = "  ".join(
    f"`{CLASS_ICONS[c]} {c.title()}`" for c in CLASS_NAMES
)
st.markdown(f"<div style='text-align:center'>{cat_display}</div>",
            unsafe_allow_html=True)

# Stats bar
st.markdown("")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Classes", "6")
with col2:
    st.metric("Training Images", "14K+")
with col3:
    st.metric("Architecture", "CNN")
with col4:
    st.metric("Inference", "<1s")
st.markdown("")

model = load_model()

# --- Upload section ---
st.markdown("### Try It Out")
uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG. Max 200MB.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col_result:
        with st.spinner("Classifying..."):
            img_array = preprocess_image(image)
            predictions = model.predict(img_array, verbose=0)[0]

        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx])
        icon = CLASS_ICONS.get(predicted_class, "")

        # Prediction card
        st.markdown(
            f'<div class="prediction-card">'
            f'<h2>{icon} {predicted_class.title()}</h2>'
            f'<p>{confidence * 100:.1f}% confidence</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Probability bars
    st.markdown("---")
    st.markdown("##### Class Probabilities")

    # Sort by probability descending
    sorted_indices = np.argsort(predictions)[::-1]

    # Colour gradient from high to low confidence
    bar_colours = ["#667eea", "#7c6dd8", "#9061c2", "#a855a0", "#c0487e", "#d63b5e"]

    for rank, idx in enumerate(sorted_indices):
        cls = CLASS_NAMES[idx]
        prob = float(predictions[idx])
        pct = prob * 100
        icon_str = CLASS_ICONS.get(cls, "")
        colour = bar_colours[rank]

        st.markdown(
            f'<div class="prob-bar-container">'
            f'  <div class="prob-label">'
            f'    <span>{icon_str} {cls.title()}</span>'
            f'    <span><strong>{pct:.1f}%</strong></span>'
            f'  </div>'
            f'  <div class="prob-bar">'
            f'    <div class="prob-fill" style="width:{pct}%; background:{colour};"></div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

else:
    # Empty state with example use cases
    st.markdown("")
    st.info("👆 Upload an image above to get started.")
    st.markdown("")

    st.markdown("### Use Cases")
    uc1, uc2, uc3 = st.columns(3)
    with uc1:
        st.markdown(
            "**🗺️ Geospatial Analysis**\n\n"
            "Automatically tag satellite and aerial imagery for land-use mapping "
            "and environmental monitoring projects."
        )
    with uc2:
        st.markdown(
            "**🏙️ Urban Planning**\n\n"
            "Classify street-level imagery at scale to support infrastructure "
            "assessment and city development initiatives."
        )
    with uc3:
        st.markdown(
            "**📸 Photo Organisation**\n\n"
            "Sort thousands of travel or survey photos into clean categories "
            "without manual effort."
        )

    st.markdown("---")
    st.markdown(
        "### Frequently Asked Questions"
    )
    with st.expander("What image formats are supported?"):
        st.write("JPG, JPEG, and PNG files up to 200MB.")
    with st.expander("How accurate is the model?"):
        st.write(
            "The model achieves over 85% accuracy on the Intel Image Classification "
            "test set across all six scene categories."
        )
    with st.expander("Is my data stored anywhere?"):
        st.write(
            "No. Images are processed in-memory and discarded immediately after "
            "classification. Nothing is saved or logged."
        )
    with st.expander("Can I use this for commercial purposes?"):
        st.write(
            "This is an academic demonstration. The underlying model and dataset "
            "(Intel Image Classification) are intended for research and educational use."
        )

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer">'
    '2026 SceneVision AI -- Intel Image Classification Demo<br>'
    'DLOR Part 2 | 2402485C Ryan Fahrein<br>'
    'Powered by TensorFlow & Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
