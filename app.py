# app.py
import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps

# Load model
MODEL_PATH = "mnist_model.h5"
@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Preprocessing function
def preprocess_image(img):
    # Convert to grayscale
    img = img.convert("L")
    # Invert colors (MNIST digits are white on black)
    img = ImageOps.invert(img)
    # Resize to 28x28
    img = img.resize((28, 28))
    # Convert to array and normalize
    arr = np.array(img).astype("float32") / 255.0
    # Heuristic inversion fix (if background white)
    if arr.mean() > 0.7:
        arr = 1.0 - arr
    # Flatten to (1,784)
    arr = arr.reshape(1, 784)
    return arr

# --- Streamlit UI ---
st.title("🧠 Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) and the model will predict it.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Predict button
    if st.button("Predict Digit"):
        x = preprocess_image(img)
        pred = model.predict(x)
        predicted_digit = np.argmax(pred, axis=1)[0]
        st.subheader(f"Predicted Digit: **{predicted_digit}**")
        st.bar_chart(pred[0])
