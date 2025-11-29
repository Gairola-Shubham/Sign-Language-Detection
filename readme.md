# 🤟 Sign Language Detection (ASL A–Z)  
### Real-Time Hand Gesture Recognition using Computer Vision + Deep Learning  
**Built with TensorFlow · Keras · OpenCV · Mediapipe · Tkinter GUI**

---

## 📌 Project Overview  
This project performs **real-time American Sign Language (ASL) alphabet detection (A–Z)** using a combination of:

- **TensorFlow/Keras** (MobileNetV2 classifier)  
- **OpenCV** for webcam stream  
- **Mediapipe** for accurate hand-detection & auto-cropping  
- **Tkinter GUI** for desktop use  
- **Custom prediction logic** to avoid accidental characters  

The system recognizes:

✔ Letters **A–Z**  
✔ **Space**  
✔ **Delete**  
✔ **Nothing** (no hand detected)  
✔ Updates text intelligently (adds delay, avoids spam letters)

---

## 🚀 Features  

### 🟣 Real-Time Detection  
- Fast, lightweight MobileNetV2 (TensorFlow)  
- Mediapipe-based **hand localization**  
- Only extracts prediction from cropped hand region  

### 🟣 Smart Prediction Logic  
- Writes a character **only when confidence is high**  
- Adds a **delay timer** to prevent rapid repeated letters  
- Ignores predictions if:  
  - No hand detected  
  - Low confidence  
  - Wrong gesture detected momentarily  

### 🟣 Full Desktop GUI  
- Live webcam feed  
- Detected character preview  
- Word builder textbox  
- Guide image for selected sign  
- Buttons: **Delete**, **Clear**  
- Clickable **A–Z, space, del** buttons (manual input)

### 🟣 Model  
- Backbone: **MobileNetV2**  
- Dataset: **ASL Alphabet Dataset (~87k images)**  
- Classes: **29 (A–Z + space + del + nothing)**  
- Achieved: **~99% validation accuracy**

---

## 📁 Project Structure

