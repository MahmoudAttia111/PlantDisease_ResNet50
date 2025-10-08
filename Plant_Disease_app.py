import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

# ==============================
# Download model from Google Drive (only if not exists locally)
# ==============================
MODEL_PATH = "best_model_resnet50.keras"
DRIVE_URL = "https://drive.google.com/uc?id=1tV_ED7daNGv_BdXi1s7QUQ5unfFoaqpN"
if not os.path.exists(MODEL_PATH):
    with st.spinner("⬇️ Downloading model from Google Drive..."):
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# ==============================
# Load the trained model
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)

# ==============================
# Load class names (hardcoded since dataset not included)
# ==============================
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="🌱 Plant Disease Classifier", layout="centered")

st.title("🌱 Plant Disease Classification")
st.write("Upload a plant leaf image and the model will predict the disease (or if it's healthy).")

# Upload image
uploaded_file = st.file_uploader("Upload a leaf image (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    # Show result
    st.subheader(f"✅ Prediction: {class_names[predicted_class]}")
    st.write(f"⚡ Confidence: {confidence:.2f}%")
