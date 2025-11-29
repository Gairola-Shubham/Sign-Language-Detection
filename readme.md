🎯 Sign Language Detection (A–Z) – Real-Time Computer Vision + Deep Learning

A real-time American Sign Language (ASL) alphabet recognition system built using:

TensorFlow / Keras

OpenCV for webcam capture

MediaPipe for hand detection + auto-cropping

Tkinter desktop GUI for interaction

MobileNetV2 deep learning model

The system predicts A–Z, Space, Delete, builds words live, and shows a guide image for each sign.

🚀 Demo

(Add your own screenshot or GIF here)

![Demo](assets/demo.gif)

⭐ Features
🔍 Real-Time Detection

Live webcam feed

MediaPipe-based hand detection

Auto hand-cropping

Ignores frames when no hand is detected

🧠 Deep Learning Model

MobileNetV2 backbone

Trained on ASL Alphabet Dataset (~87k images)

Supports 29 classes

A–Z

space

del

nothing (internal logic only)

Achieved ~99% validation accuracy

📝 Smart Prediction Logic

Writes a letter only when a hand is detected

Avoids duplicate characters by smoothing predictions

Add / delete characters in real time

Built-in guide displaying the reference sign for each letter

🖼️ Project Screenshot

(Add your GUI screenshot here)

![GUI](assets/gui.png)

📂 Project Structure
├── app.py               # Basic prediction script
├── app_gui.py           # Full Tkinter GUI application
├── train.py             # Model training script
├── prep_split.py        # Dataset preparation
├── artifacts/
│   ├── best_model.h5
│   ├── class_indices.json
│   └── preprocess.json
├── guide/               # A.jpg, B.jpg ... Z.jpg reference images
├── data/                # Ignored (train/val/test)
├── readme.md
└── requirement.txt

🔧 Installation
1️⃣ Clone the repository
git clone https://github.com/Gairola-Shubham/Sign-Language-Detection.git
cd Sign-Language-Detection

2️⃣ Create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirement.txt

▶️ Run the Application
Start the GUI
python app_gui.py

Or run the basic version
python app.py

📦 Dataset

This project uses the ASL Alphabet Dataset (Kaggle)
Dataset is NOT included in the repo due to size.

Download from:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

🧪 Training the Model

If you want to retrain the model:

Place the dataset in:

asl_alphabet_train/
asl_alphabet_test/


Run:

python prep_split.py
python train.py


Model & metadata will be saved to the artifacts/ folder.

🛠️ Tech Stack
Component	Technology
Deep Learning	TensorFlow / Keras
Computer Vision	OpenCV
Hand Detection	MediaPipe
Model Backbone	MobileNetV2
GUI	Tkinter
Dataset	ASL Alphabet Dataset
🤝 Contributing

Contributions are welcome!
Feel free to submit issues or pull requests.
