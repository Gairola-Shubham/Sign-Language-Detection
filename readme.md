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
```text
Sign-Language-Detection/
│
├── app.py                 # Basic prediction (console)
├── app_gui.py             # Full Tkinter GUI Application
├── train.py               # Model training script
├── prep_split.py          # Train/Val/Test splitter
│
├── artifacts/
│   ├── best_model.h5      # Trained model
│   ├── class_indices.json # Index → class mapping
│   └── preprocess.json    # Preprocessing config
│
├── asl_alphabet_train/    # Original train dataset 
├── asl_alphabet_test/
│
│   ├── train/
│   ├── val/
│   └── test/
│
├── guide/                 # A.jpg, B.jpg ... Z.jpg reference images
│
├── requirements.txt
└── README.md
```
## 🔧 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Gairola-Shubham/Sign-Language-Detection.git
cd Sign-Language-Detection
2️⃣ Create Virtual Environment
bash
Copy code
python -m venv .venv
.venv\Scripts\activate   # Windows
3️⃣ Install Requirements
bash
Copy code
pip install -r requirements.txt
▶️ Run the Application
GUI Version
bash
Copy code
python app_gui.py
Basic Version
bash
Copy code
python app.py
📦 Dataset (Not Included)
This project uses the ASL Alphabet Dataset from Kaggle:

🔗 https://www.kaggle.com/datasets/grassknoted/asl-alphabet

After downloading, place these folders:

Copy code
asl_alphabet_train/
asl_alphabet_test/
🏋️ Training the Model
Split dataset into train/val/test
bash
Copy code
python prep_split.py
Train the Model
bash
Copy code
python train.py
The trained model will be saved at:

bash
Copy code
artifacts/best_model.h5
🛠 Tech Stack
Python 3.10–3.11

TensorFlow / Keras

OpenCV

Mediapipe

Tkinter

NumPy

🎯 Future Enhancements
Two-hand recognition

Add numbers (0–9)

Sentence-level prediction

Deploy as a web app

