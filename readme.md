Sign Language Detection (ASL A–Z) – Real-Time ML + Computer Vision Project

A real-time American Sign Language (ASL) alphabet recognition system built using:

TensorFlow / Keras

OpenCV for video capture

Mediapipe for hand detection

Tkinter GUI for desktop application

The system recognizes A–Z, Space, and Del, builds words in real time, and shows a guide image for each sign.

🚀 Features
✔ Real-Time Detection

Mediapipe-based hand detection

Auto hand cropping

Ignores frames when no hand is detected

✔ Deep Learning Model

MobileNetV2 backbone

Trained on ASL Alphabet Dataset (~87k images)

29 classes (A–Z + space + del + nothing)

~99% validation accuracy

✔ Smart Prediction Logic

Confidence thresholding

Temporal smoothing to avoid flicker

Only types when detection is stable

✔ Clean Desktop GUI

Live webcam feed

Real-time prediction and confidence

Text builder (typing output)

Delete & Clear buttons

Grid of guide images

Fully responsive layout

🧠 Model Artifacts

Saved model components:

artifacts/
│── best_model.h5
│── class_indices.json
└── preprocess.json

📁 Project Structure
Sign-Language-Detection/
│
├── app_gui.py
├── app.py                (optional, CLI version)
├── requirements.txt
├── README.md
│
├── artifacts/
│   ├── best_model.h5
│   ├── class_indices.json
│   └── preprocess.json
│
└── guide/
    ├── A.jpg
    ├── B.jpg
    ├── ...
    └── space.jpg

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt

2. Run the GUI application
python app_gui.py

🛠 Technologies Used

Python

TensorFlow / Keras

OpenCV

Mediapipe

Tkinter

NumPy

📌 Future Improvements

Support for two-hand ASL gestures

Continuous word/sentence recognition

Noise-resistant tracking

Mobile version using TFLite

Webcam auto-exposure & brightness control
