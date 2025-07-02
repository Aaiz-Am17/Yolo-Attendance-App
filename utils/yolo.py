from ultralytics import YOLO
import os

def train_face_detector():
    data_yaml = "datasets/detector/data.yaml"  # Path to your data.yaml for face detection
    output_dir = "models/face_detector_training"

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO("yolov5s.pt")

    model.train(
        data=data_yaml,
        epochs=300,
        imgsz=640,
        batch=16,
        lr0=0.01,
        workers=4,
        patience=50,
        device='cpu',  # Change to '0' if using GPU
        project=output_dir,
        name="YOLOv5_Face_Detector"
    )

    print("✅ Detection training complete!")

if __name__ == "__main__":
    train_face_detector()
