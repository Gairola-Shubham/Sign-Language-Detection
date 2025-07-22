from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import logging
import os
import sys

app = Flask(__name__)

# ✅ Robust model path check
MODEL_DIR = os.path.join(os.getcwd(), 'trained_model')
MODEL_FILENAME = 'signlanguagedetectionmodel50x50.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Model file not found at {MODEL_PATH}")
    sys.exit(1)  # Exit if model file doesn't exist

model = load_model(MODEL_PATH)

# Labels corresponding to the model output
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['blank']

# Normalize and reshape image for prediction
def preprocess_image(image):
    image = np.array(image)
    image = image.reshape(1, 50, 50, 1)
    return image / 255.0

# Initialize webcam
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            app.logger.error("Failed to capture frame")
            break
        else:
            # Draw rectangle and crop region
            cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 2)
            cropped = frame[40:300, 0:300]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (50, 50))
            preprocessed = preprocess_image(resized)

            # Predict using model
            predictions = model.predict(preprocessed)
            predicted_label = labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            # Display prediction on frame
            display_text = f"{predicted_label} ({confidence:.2f}%)" if predicted_label != "blank" else " "
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
