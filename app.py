import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

# === CONFIG ===
MODEL_PATH = "src/models/MobileNetV2_model.h5"
IMG_SIZE = (128, 128)
CLASS_NAMES = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']

# === Load Model ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# === Streamlit UI ===
st.title("🧠 Alzheimer MRI Classification")
st.write("Upload an MRI scan to predict the dementia stage.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=IMG_SIZE)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    confidence = np.max(predictions) * 100
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    st.subheader("🧠 Prediction:")
    st.write(f"**{predicted_class}** ({confidence:.2f}% confidence)")

    # Optional: ground truth label
    gt_label = st.selectbox("Select actual label (optional):", ["None"] + CLASS_NAMES)
    if gt_label != "None":
        if gt_label == predicted_class:
            st.success("✅ Correct prediction!")
        else:
            st.error(f"❌ Incorrect prediction. Actual: {gt_label}")
