import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import os
import gdown

# -------------------------
# Model configuration
# -------------------------
MODEL_PATH = 'best_model_tf.keras'
FILE_ID = '1jfr9gX1YhZiEafXq3pVlwQHF6NN-qn-F'
GOOGLE_DRIVE_LINK = f'https://drive.google.com/uc?id={FILE_ID}'

# Class labels (in the exact order your model was trained)
class_names = [
    'cardboard', 'e-waste', 'glass', 'metal', 'organic',
    'paper', 'plastic', 'textile', 'trash'
]

# -------------------------
# Download model from Google Drive
# -------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found locally. Downloading from Google Drive...")
        try:
            gdown.download(GOOGLE_DRIVE_LINK, MODEL_PATH, quiet=False)
            if os.path.getsize(MODEL_PATH) < 1 * 1024 * 1024:  # Check if < 1MB
                raise Exception("Model file is too small. Likely corrupted.")
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            raise e
    else:
        st.info("✅ Model already exists locally.")

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# -------------------------
# Preprocess uploaded image
# -------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# -------------------------
# Predict function
# -------------------------
def predict(model, img_array):
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = preds[0][class_idx]
    return class_names[class_idx], confidence, preds[0]

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Waste Classifier", layout="centered")
    st.title("♻️ Waste Classification App")
    st.markdown("Upload a waste image and let the model classify it into one of the 9 categories.")

    # Step 1: Ensure model is downloaded
    download_model()

    # Step 2: Load the model
    try:
        model = load_model()
    except Exception as e:
        st.error("❌ Failed to load the model.")
        st.stop()

    # Step 3: Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(img)
        class_name, confidence, all_probs = predict(model, img_array)

        st.markdown(f"### 🧠 Prediction: **{class_name}**")
        st.markdown(f"### 📊 Confidence: **{confidence * 100:.2f}%**")

        # Show all class probabilities
        st.subheader("Class Probabilities")
        for i, prob in enumerate(all_probs):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")
            st.progress(float(prob))

if __name__ == "__main__":
    main()
