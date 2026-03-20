import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="centered",
)

# -------------------------------------------------
# Constants
# -------------------------------------------------
MODEL_NAME = "EfficientNetB0"
IMAGE_SIZE = (224, 224)
TOP_K = 5
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

# -------------------------------------------------
# Load model once and cache it
# -------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.applications.EfficientNetB0(weights="imagenet")

model = load_model()

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def preprocess_image(uploaded_file) -> tuple[np.ndarray, Image.Image]:
    """Read uploaded image, convert to RGB, resize, and return batch tensor."""
    image = Image.open(uploaded_file).convert("RGB")
    display_image = image.copy()

    image = image.resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32)

    # EfficientNet in current Keras includes preprocessing internally.
    batch = np.expand_dims(arr, axis=0)
    return batch, display_image


def predict_image(batch: np.ndarray):
    preds = model.predict(batch, verbose=0)
    decoded = tf.keras.applications.efficientnet.decode_predictions(preds, top=TOP_K)[0]
    return [
        {
            "imagenet_id": imagenet_id,
            "label": label.replace("_", " "),
            "confidence": float(score),
        }
        for imagenet_id, label, score in decoded
    ]


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🖼️ Image Classification App")
st.write(
    "Upload an image and the app will predict the most likely ImageNet classes "
    "using EfficientNetB0."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=SUPPORTED_TYPES,
    help="Supported formats: jpg, jpeg, png, bmp, webp",
)

if uploaded_file is not None:
    st.subheader("Uploaded image")
    st.write(f"**Filename:** {uploaded_file.name}")

    try:
        batch, display_image = preprocess_image(uploaded_file)
        st.image(display_image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Running classification..."):
            predictions = predict_image(batch)

        st.subheader("Prediction summary")
        top_pred = predictions[0]
        st.success(
            f"Top prediction: **{top_pred['label']}** "
            f"({top_pred['confidence']:.2%} confidence)"
        )

        st.subheader("Top 5 predictions")
        for i, pred in enumerate(predictions, start=1):
            st.write(f"{i}. **{pred['label']}** — {pred['confidence']:.2%}")
            st.progress(min(max(float(pred["confidence"]), 0.0), 1.0))

        st.subheader("Raw JSON-style output")
        st.json(
            {
                "filename": uploaded_file.name,
                "model": MODEL_NAME,
                "input_size": list(IMAGE_SIZE),
                "prediction_summary": {
                    "top_prediction": top_pred["label"],
                    "confidence": top_pred["confidence"],
                },
                "predictions": predictions,
            }
        )

    except Exception as e:
        st.error(f"Could not process the uploaded file: {e}")

st.markdown("---")
st.caption(
    "Tip: For Streamlit Community Cloud, place this file in your GitHub repo as "
    "`app.py` and set the app entrypoint to `app.py`."
)
