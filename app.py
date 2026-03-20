import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="centered",
)

MODEL_NAME = "EfficientNetB0"
IMAGE_SIZE = (224, 224)
TOP_K = 5
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

@st.cache_resource
def load_model():
    return tf.keras.applications.EfficientNetB0(weights="imagenet")

model = load_model()

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    show_image = image.copy()
    image = image.resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    return batch, show_image

def predict_image(batch):
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

st.title("🖼️ Image Classification App")
st.write("Upload an image and the app will classify it with EfficientNetB0.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=SUPPORTED_TYPES,
    help="Supported file types: jpg, jpeg, png, bmp, webp"
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    try:
        batch, show_image = preprocess_image(uploaded_file)

        with st.spinner("Running classification..."):
            predictions = predict_image(batch)

        top_pred = predictions[0]

        st.subheader("Prediction summary")
        st.success(
            f"Top prediction: {top_pred['label']} "
            f"({top_pred['confidence']:.2%} confidence)"
        )

        st.subheader("Top 5 predictions")
        for i, pred in enumerate(predictions, start=1):
            st.write(f"{i}. {pred['label']} — {pred['confidence']:.2%}")
            st.progress(min(max(pred["confidence"], 0.0), 1.0))

        st.subheader("JSON output")
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

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image.")
    except Exception as e:
        st.error(f"Error while processing the image: {e}")

st.markdown("---")
st.caption("Deploy this file as app.py on Streamlit Community Cloud.")
