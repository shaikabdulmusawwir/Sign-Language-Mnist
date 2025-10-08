import os
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# Automatically find model path relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sign_model.h5")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

# ------------------------------------------------------------
# LOAD MODEL (cached to avoid reloading each time)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ------------------------------------------------------------
# STREAMLIT APP INTERFACE
# ------------------------------------------------------------
st.title("🤟 Sign Language MNIST Classifier")
st.write("Upload a hand sign image (28x28 pixels) to predict the corresponding alphabet (A–Z).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display image
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = np.array(image.resize((28, 28))) / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)

    # Convert number to corresponding alphabet
    labels = [chr(i + 65) for i in range(26)]  # ['A', 'B', ... 'Z']
    predicted_char = labels[predicted_label]

    st.success(f"✅ Predicted Sign: **{predicted_char}**")

    # Show probabilities (optional)
    st.write("Prediction probabilities:")
    st.bar_chart(prediction[0])
else:
    st.info("Please upload an image to get a prediction.")
