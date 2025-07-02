🧠 Attendance AI – YOLO-Powered Face Detection & Recognition
Real-Time Face Detection and Classification for Smart Attendance Systems
GitHub • Last Commit
License: MIT • Python • Built with Streamlit + YOLOv5 + YOLOv8

💡 Project Overview & Motivation
Attendance AI is a web-based application that combines the power of YOLOv5 (for face detection) and YOLOv8 (for face classification) to recognize people in real-time via uploaded images or webcam. Built using Streamlit, it features an intuitive UI and modular backend for fast, deployable AI-based attendance tracking.

The idea was born from a simple need — streamlining attendance in labs, classrooms, or small teams using deep learning, with a fun, engaging interface.
Rather than scan barcodes or mark roll calls manually, what if you could just take a picture?

This system detects who’s in the image and lists their names with confidence scores — all powered by your own trained YOLO models!

✨ Key Features
🔍 Real-Time Face Detection
Utilizes YOLOv5 to locate multiple faces in a frame with precise bounding boxes.

🧠 Identity Classification
Cropped faces are passed through a YOLOv8 classifier to predict identities based on your trained dataset.

📸 Upload or Webcam Input
Use either uploaded images or live webcam to detect and classify faces.

🖥️ Streamlit-Based GUI
Modern, dark-mode web app for fast testing and demo — no CLI needed!

🗃️ Modular Training Scripts
Easily retrain detection and classification models on your own dataset using included scripts.

🚀 How It Works
Upload an Image or Use Webcam

Faces are detected using YOLOv5

Cropped faces are passed to a YOLOv8 classifier

The app displays predicted names with confidence scores

📁 Project Structure
pgsql
Copy
Edit
YOLO-Attendance-App/
├── app.py                  # Streamlit web app (main UI)
├── utils/
│   ├── AI_Model.py         # Preprocessing: JPG conversion, upscaling, face cropping
│   ├── classifier.py       # Train YOLOv8 classifier
│   └── yolo.py             # Train YOLOv5 face detector
├── models/
│   ├── face_detector.pt    # Pretrained YOLOv5 model (included)
│   └── classifier.pt       # Classification model (not public)
├── temp/                   # Temporary image storage
├── requirements.txt        # Project dependencies
├── LICENSE                 # MIT License
└── README.md               # You are here
📦 Pretrained Weights
✅ Face Detection Model (face_detector.pt)
Trained using YOLOv5 on custom face dataset.
📁 Location: models/face_detector.pt

🔒 Face Classification Model (classifier.pt)
Not included due to privacy of training data (private student faces).

🛠️ Setup and Installation
Clone the Repository

bash
Copy
Edit
git clone https://github.com/yourusername/YOLO-Attendance-App.git
cd YOLO-Attendance-App
Create Virtual Environment (Recommended)

bash
Copy
Edit
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the App

bash
Copy
Edit
streamlit run app.py
🧪 Optional: Training Your Own Models
You can retrain both YOLO models using your own image datasets:

🟣 YOLOv5 Face Detection:
Use utils/yolo.py and a data.yaml with face annotations

🔵 YOLOv8 Classification:
Organize your face dataset in datasets/classifier/train/class_name/*.jpg format and run utils/classifier.py

🙋‍♂️ Contributing
Pull requests are welcome! If you have ideas for improvements, please:

Fork the repo

Create a branch

Make your changes

Submit a PR with a clear description

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

👥 Credits
Developed by Aaiz Mohsin (BS AI, GIKI) as part of the Deep Neural Networks course project (Semester 5).
Special thanks to Ultralytics for YOLO.

🧠 Let's Talk AI
Got feedback or ideas?
Connect on LinkedIn or drop a ⭐ on GitHub!
