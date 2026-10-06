import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ---------------------------------------
# Page configuration
# ---------------------------------------

st.set_page_config(
    page_title="Early Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# ---------------------------------------
# Title
# ---------------------------------------

st.title("🌱 Early Plant Disease Detection")

st.write(
    "Upload a plant leaf image and the system will "
    "predict the possible disease."
)

# ---------------------------------------
# Load model
# ---------------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "plant_disease_model.keras"
    )

    return model


# ---------------------------------------
# Load class names
# ---------------------------------------

@st.cache_data
def load_class_names():

    with open("class_names.json", "r") as file:
        class_names = json.load(file)

    return class_names


# Load model and classes

model = load_model()
class_names = load_class_names()


# ---------------------------------------
# Upload image
# ---------------------------------------

uploaded_file = st.file_uploader(
    "📷 Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)


# ---------------------------------------
# Prediction
# ---------------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # -----------------------------------
    # Display uploaded image
    # -----------------------------------

    st.subheader("📷 Uploaded Leaf")

    st.image(
        image,
        caption="Uploaded Plant Leaf",
        width="stretch"
    )

    # -----------------------------------
    # Image processing
    # -----------------------------------

    processed_image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        processed_image
    )

    # Convert image to batch
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # -----------------------------------
    # Prediction
    # -----------------------------------

    with st.spinner(
        "🔍 Analyzing the leaf..."
    ):

        prediction = model.predict(
            image_array,
            verbose=0
        )

    # Find highest probability
    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_disease = class_names[
        predicted_index
    ]

    confidence = (
        prediction[0][predicted_index]
        * 100
    )

    # -----------------------------------
    # Display result
    # -----------------------------------

    st.subheader(
        "🔍 Detection Result"
    )

    st.success(
        f"🌿 Disease Detected: {predicted_disease}"
    )

    st.info(
        f"🎯 Confidence: {confidence:.2f}%"
    )

    # -----------------------------------
    # Show all probabilities
    # -----------------------------------

    st.subheader(
        "📊 Prediction Probabilities"
    )

    for i, disease in enumerate(class_names):

        probability = (
            prediction[0][i] * 100
        )

        st.write(
            f"{disease}: {probability:.2f}%"
        )

        st.progress(
            int(probability)
        )