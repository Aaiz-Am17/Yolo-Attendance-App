import os
from ultralytics import YOLO

def train_yolo_classifier():
    dataset_dir = "datasets/classifier"  # Should contain train/val folders
    model_save_dir = "models/classifier_training"

    os.makedirs(model_save_dir, exist_ok=True)
    model = YOLO("yolov8n-cls.pt")  # Start from pretrained classification model

    model.train(
        data=dataset_dir,
        epochs=100,
        imgsz=448,
        batch=16,
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0001,
        dropout=0.2,
        cos_lr=True,
        amp=True,
        patience=20,
        project=model_save_dir,
        name="YOLOv8_Classifier"
    )

    print("✅ Training complete!")

if __name__ == "__main__":
    train_yolo_classifier()
