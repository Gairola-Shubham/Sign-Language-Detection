🤟 Sign Language Detection (ASL A–Z)
Real-Time Hand Gesture Recognition using Computer Vision + Deep Learning

Built with TensorFlow · Keras · OpenCV · Mediapipe · Tkinter GUI

📌 Project Overview

This project performs real-time American Sign Language (ASL) alphabet detection (A–Z) using:

TensorFlow/Keras (MobileNetV2 classifier)

OpenCV for webcam processing

Mediapipe for hand detection & auto-cropping

Tkinter GUI for user interface

Smart prediction logic to avoid false or repeated letters

The system detects:

✔ A–Z
✔ space
✔ del
✔ nothing (no hand detected)
✔ Writes characters only when prediction is stable

🚀 Features
🔍 Real-Time Hand Detection

Uses Mediapipe for accurate hand tracking

Automatically crops only the hand area

Ignores prediction when hand is not visible

🧠 Deep Learning Model

MobileNetV2-based classifier

Trained on ASL Alphabet Dataset (~87K images)

29 classes total (A–Z, space, del, nothing)

~99% validation accuracy

✍ Smart Prediction System

Confidence thresholding

Delay timer to prevent duplicate typing

Smooth frame-to-frame prediction

Delete & Clear buttons

🖥 GUI Application

Live webcam preview

Real-time predicted letter

Word builder textbox

Guide image for each letter

A–Z buttons for manual input

📁 Project Structure
Sign-Language-Detection/
│
├── app.py                 # Basic prediction script (CLI)
├── app_gui.py             # Full Tkinter GUI
├── train.py               # Model training script
├── prep_split.py          # Dataset splitter
│
├── artifacts/
│   ├── best_model.h5
│   ├── class_indices.json
│   └── preprocess.json
│
├── asl_alphabet_train/    # (Optional) Original Kaggle train dataset
├── asl_alphabet_test/     # (Optional) Original Kaggle test dataset
│
├── data/                  # Auto-generated train/val/test split
│   ├── train/
│   ├── val/
│   └── test/
│
├── guide/                 # A.jpg, B.jpg … Z.jpg
│
├── requirements.txt
└── README.md

🔧 Installation
1️⃣ Clone the Repository
git clone https://github.com/Gairola-Shubham/Sign-Language-Detection.git
cd Sign-Language-Detection

2️⃣ Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

3️⃣ Install Requirements
pip install -r requirements.txt

▶️ Running the Application
Run GUI (Recommended):
python app_gui.py

Run simple mode:
python app.py

📦 Dataset (Not Included in Repo)

Dataset used: ASL Alphabet Dataset (Kaggle)
Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Expected folders:

asl_alphabet_train/
asl_alphabet_test/


Split dataset into train/val/test:

python prep_split.py

🧠 Training the Model

To retrain:

python train.py


Artifacts saved in:

artifacts/
  best_model.h5
  class_indices.json
  preprocess.json

🔤 ASL Alphabet Guide (Placeholders)
A	B	C
(A.jpg)	(B.jpg)	(C.jpg)
D	E	F
(D.jpg)	(E.jpg)	(F.jpg)
G	H	I
(G.jpg)	(H.jpg)	(I.jpg)
J	K	L
(J.jpg)	(K.jpg)	(L.jpg)
M	N	O
(M.jpg)	(N.jpg)	(O.jpg)
P	Q	R
(P.jpg)	(Q.jpg)	(R.jpg)
S	T	U
(S.jpg)	(T.jpg)	(U.jpg)
V	W	X
(V.jpg)	(W.jpg)	(X.jpg)
Y	Z	Space / Del
(Y.jpg)	(Z.jpg)	(space/del)
🔮 Future Improvements

Two-hand gesture support

Motion tracking for letters like J and Z

NLP-based sentence prediction

Mobile app version using TFLite

Gesture customization and personal calibration
