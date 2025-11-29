# 🤟 Sign Language Detection (ASL A–Z)
### Real-Time Hand Gesture Recognition using Computer Vision + Deep Learning  
**Built with:** TensorFlow · Keras · OpenCV · Mediapipe · Tkinter

---

## 📌 Overview
This project performs **real-time American Sign Language (ASL) alphabet detection** using machine learning and computer vision.

It recognizes:

- **A–Z**
- **Space**
- **Delete**
- **Nothing** (no hand detected)

The system auto-crops hands, ignores bad frames, and builds words using a smart delay logic.

---

## 🚀 Features

### 🔍 Real-Time Detection
- Live webcam feed  
- Mediapipe-based hand detection  
- Auto-crop hand region  
- Skips frames when no hand is detected  

### 🧠 Deep Learning Model
- MobileNetV2 CNN  
- Trained on ASL Alphabet Dataset (~87k images)  
- **29 classes** (A–Z + space + del + nothing)  
- ~**99% validation accuracy**

### 🖥 GUI (Tkinter)
- Large camera display  
- Live prediction + confidence  
- Word-builder text box  
- Guide image for each letter  
- A–Z buttons for manual input  
- Delete and Clear buttons  

---

## 📁 Project Structure
<pre> ``` Sign-Language-Detection/ │ ├── app.py # Basic prediction (console) ├── app_gui.py # Full Tkinter GUI Application ├── train.py # Model training script ├── prep_split.py # Train/Val/Test splitter │ ├── artifacts/ │ ├── best_model.h5 # Trained model │ ├── class_indices.json # Index → class mapping │ └── preprocess.json # Preprocessing config │ ├── asl_alphabet_train/ ├── asl_alphabet_test/ │ │ ├── train/ │ ├── val/ │ └── test/ │ ├── guide/ # Reference images (A–Z) │ ├── requirements.txt └── README.md ``` </pre>
