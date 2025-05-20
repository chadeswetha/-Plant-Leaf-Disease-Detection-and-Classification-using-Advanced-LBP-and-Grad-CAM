import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
import cv2
import matplotlib.pyplot as plt

# Convert PIL image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode()

# Simple heuristic to check if image contains green (leaf-like)
def is_leaf_image(image):
    img_resized = image.resize((100, 100))
    img_np = np.array(img_resized)

    # Convert to HSV to detect green
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (100 * 100)

    return green_ratio > 0.2  # 20% of image must be green

# Load model
MODEL_PATH = "plant_disease_model.h5"
model = load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = [
    'Bacterial_spot', 'healthy', 'Late_blight','Leaf_Mold','YellowLeaf_Curl_Virus',
    'spotted_spider_mite','Septoria_leaf_spot', 'healthy',
    'Spider_mites_Two_spotted_spider_mite', 'Early_blight', 'Target_Spot',
    'healthy', 'Late_blight', 'Early_blight', 'mosaic_virus'
]

# Page configuration
st.set_page_config(
    page_title="AgroLeaf — Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# 💅 Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f3fff5;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-container {
            background: #ffffff;
            padding: 2rem;
            border-radius: 18px;
            box-shadow: 0 6px 25px rgba(0, 128, 0, 0.15);
            margin-top: 30px;
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
        }
        .stButton>button {
            background: linear-gradient(to right, #228B22, #32CD32);
            color: white;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #32CD32, #7CFC00);
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .uploaded-img {
            text-align: center;
            margin-top: 20px;
            animation: fadeIn 0.8s ease;
            width: 40%;
        }
        .uploaded-img img {
            border-radius: 15px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.1);
            width: 100%;
        }
        .uploaded-img p {
            color: #2e8b57;
            font-weight: bold;
            margin-top: 10px;
        }
        h1 {
            color: #006400;
            font-size: 36px;
            text-align: center;
            margin-bottom: 20px;
            flex-grow: 1;
        }
        .confidence-bar {
            height: 20px;
            border-radius: 10px;
            background: #e0ffe6;
            overflow: hidden;
            margin-top: 10px;
        }
        .confidence-bar-inner {
            height: 100%;
            background: #32CD32;
            text-align: right;
            color: white;
            padding-right: 8px;
            line-height: 20px;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# 💡 Main App
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("🌿 AgroLeaf: Plant Disease Detector")
st.markdown("Upload a plant leaf image below to detect potential disease using our AI-powered deep learning model.")

# Image upload
uploaded_file = st.file_uploader("📸 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    encoded_img = image_to_base64(img)

    st.markdown(
        f"""
        <div class='uploaded-img'>
            <img src="data:image/png;base64,{encoded_img}" width="280"/>
            <p>Uploaded Leaf Image</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("🔍 Predict"):
        if not is_leaf_image(img):
            st.error("🚫 The uploaded image doesn't appear to be a **leaf**. Please upload a valid leaf image.")
        else:
            img = img.resize((256, 256)).convert("RGB")
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions[0])
            predicted_label = CLASS_NAMES[predicted_index]
            confidence = predictions[0][predicted_index] * 100

            st.success(f"🌱 Prediction: **{predicted_label}**")
            st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-bar-inner" style="width:{confidence:.2f}%;">{confidence:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
