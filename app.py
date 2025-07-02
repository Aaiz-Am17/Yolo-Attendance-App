import os
import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Paths to model weights (update with your actual filenames or leave as default)
FACE_DETECTION_MODEL_PATH = "models/face_detector.pt"
CLASSIFICATION_MODEL_PATH = "models/classifier.pt"

# Constants
SCALE_FACTOR = 2
RESIZE_TO = 448
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Load models
face_detection_model = YOLO(FACE_DETECTION_MODEL_PATH)
classification_model = YOLO(CLASSIFICATION_MODEL_PATH)

def upscale_image(image, scale_factor=SCALE_FACTOR):
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def detect_and_crop_faces(image_path, detection_model):
    cropped_faces = []
    results = detection_model.predict(source=image_path, conf=0.15)
    image = cv2.imread(image_path)
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cropped_face = image[y1:y2, x1:x2]
        if cropped_face.shape[0] < 30 or cropped_face.shape[1] < 30:
            continue
        cropped_faces.append(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    return cropped_faces

def classify_faces(cropped_faces, classification_model):
    present_people = []
    for i, cropped_face in enumerate(cropped_faces):
        try:
            img = Image.fromarray(cropped_face).resize((RESIZE_TO, RESIZE_TO))
            temp_path = os.path.join(TEMP_DIR, f"resized_{i}.jpg")
            img.save(temp_path)
            results = classification_model.predict(source=temp_path, conf=0.15)
            if hasattr(results[0], 'probs') and results[0].probs is not None:
                probs = results[0].probs.cpu().numpy()
                for class_id, prob in enumerate(probs):
                    if prob > 0.01:
                        present_people.append(results[0].names[class_id])
            elif hasattr(results[0], 'top1'):
                present_people.append(results[0].names[results[0].top1])
        except Exception as e:
            st.error(f"Error during classification: {e}")
    return list(set(present_people))

# --- Streamlit UI ---
st.set_page_config(page_title="Attendance AI", layout="wide")
st.title("🧠 Attendance AI – Face Detection & Recognition")

st.sidebar.header("Choose Mode")
option = st.sidebar.radio("Select Input Method:", ["Upload Image", "Use Webcam"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_path = os.path.join(TEMP_DIR, "uploaded.jpg")
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

        cropped_faces = detect_and_crop_faces(image_path, face_detection_model)
        if cropped_faces:
            present_people = classify_faces(cropped_faces, classification_model)
            st.markdown(f"### ✅ People Detected: {', '.join(present_people)}")
        else:
            st.warning("No faces detected.")

elif option == "Use Webcam":
    st.info("Webcam mode coming soon! (or implement using `streamlit_webrtc`)")

st.markdown("---")
st.caption("Built with YOLOv5 & YOLOv8 · GIKI DNN Course Project")
