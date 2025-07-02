import os
import random
import cv2
from PIL import Image
from ultralytics import YOLO

# Step 1: Convert Images to JPG Format
def convert_images_to_jpg(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    supported_formats = ('.png', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.jpg')

    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        output_subdir = os.path.join(output_dir, folder_name)
        os.makedirs(output_subdir, exist_ok=True)

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if not file_name.lower().endswith(supported_formats):
                continue

            try:
                with Image.open(file_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    new_file_name = os.path.splitext(file_name)[0] + ".jpg"
                    save_path = os.path.join(output_subdir, new_file_name)
                    img.save(save_path, format='JPEG')
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Step 2: Upscale Images
def upscale_images(input_dir, output_dir, scale_factor=2):
    os.makedirs(output_dir, exist_ok=True)

    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        output_subdir = os.path.join(output_dir, folder_name)
        os.makedirs(output_subdir, exist_ok=True)

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                with Image.open(file_path) as img:
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)
                    upscaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    save_path = os.path.join(output_subdir, file_name)
                    upscaled_img.save(save_path, format='JPEG')
            except Exception as e:
                print(f"Error upscaling {file_path}: {e}")

# Step 3: Crop Faces and Split
def crop_faces_and_split(yolo_weights, input_dir, output_dir, train_ratio=0.8):
    model = YOLO(yolo_weights)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        images = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for img_path in train_images:
            crop_and_save_faces(model, img_path, train_class_dir)
        for img_path in val_images:
            crop_and_save_faces(model, img_path, val_class_dir)

def crop_and_save_faces(model, img_path, output_dir):
    try:
        image = cv2.imread(img_path)
        if image is None:
            return

        results = model.predict(source=img_path, conf=0.15)
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cropped_face = image[y1:y2, x1:x2]

            if cropped_face.shape[0] < 30 or cropped_face.shape[1] < 30:
                continue

            face_filename = f"{os.path.splitext(os.path.basename(img_path))[0]}_face_{i}.jpg"
            face_output_path = os.path.join(output_dir, face_filename)
            cv2.imwrite(face_output_path, cropped_face)
    except Exception as e:
        print(f"Error cropping face from {img_path}: {e}")

# Optional CLI usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLO Face Dataset Prep")
    parser.add_argument('--raw', type=str, required=True, help="Path to raw input images")
    parser.add_argument('--converted', type=str, required=True, help="Path to save JPGs")
    parser.add_argument('--upscaled', type=str, required=True, help="Path to save upscaled images")
    parser.add_argument('--cropped', type=str, required=True, help="Path to save cropped faces")
    parser.add_argument('--weights', type=str, required=True, help="Path to YOLOv5 face detection model")
    args = parser.parse_args()

    convert_images_to_jpg(args.raw, args.converted)
    upscale_images(args.converted, args.upscaled)
    crop_faces_and_split(args.weights, args.upscaled, args.cropped)
