import cv2
import numpy as np
import tensorflow as tf
import json
import time
import os
from tkinter import *
from PIL import Image, ImageTk
import mediapipe as mp

# ============================
# LOAD MODEL & METADATA
# ============================
MODEL_PATH = "artifacts/best_model.h5"
CLASS_INDEX_PATH = "artifacts/class_indices.json"
PREPROCESS_PATH = "artifacts/preprocess.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

# Fix missing labels (29 classes)
full_labels = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R",
    "S","T","U","V","W","X","Y","Z","del","nothing","space"
]
index_to_class = {i: full_labels[i] for i in range(29)}

with open(PREPROCESS_PATH, "r") as f:
    preprocess = json.load(f)

IMG_SIZE = preprocess["img_size"]

# ============================
# MEDIAPIPE HANDS
# ============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ============================
# PREDICTION CONTROL
# ============================
CONF_THRESHOLD = 0.75
REQUIRED_FRAMES = 4
COOLDOWN_TIME = 1.0

stable_letter = ""
frame_count = 0
last_time = 0

# ============================
# ML PREDICTION
# ============================
def predict_letter(crop):
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(pred[idx])

    return index_to_class[idx], conf


# ============================
# GUIDE IMAGE LOADING
# ============================
def update_guide_image(letter):
    fname = f"{letter}.jpg" if letter not in ["space", "del"] else f"{letter}.jpg"
    path = os.path.join("guide", fname)

    if not os.path.exists(path):
        guide_canvas.delete("all")
        return

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))

    imgtk = ImageTk.PhotoImage(Image.fromarray(img))
    guide_canvas.imgtk = imgtk
    guide_canvas.create_image(0, 0, anchor="nw", image=imgtk)


# ============================
# GUI
# ============================
root = Tk()
root.title("ASL Builder — GUI")
root.geometry("1500x900")
root.configure(bg="#222")

main_frame = Frame(root, bg="#222")
main_frame.pack(fill="both", expand=True)

# Left: Camera
camera_frame = Frame(main_frame, bg="black")
camera_frame.pack(side="left", fill="both", expand=True)

camera_label = Label(camera_frame, bg="black")
camera_label.pack(fill="both", expand=True)

# Right: Controls
right_frame = Frame(main_frame, bg="#222")
right_frame.pack(side="right", fill="y", padx=20)

prediction_label = Label(right_frame, text="Prediction: No hand",
                         font=("Arial", 22, "bold"), bg="#222", fg="white")
prediction_label.pack(pady=10)

output_text = Label(right_frame, text="", bg="black", fg="white",
                    font=("Arial", 26), width=20, height=2, anchor="nw")
output_text.pack()

# Delete / Clear
btn_frame = Frame(right_frame, bg="#222")
btn_frame.pack(pady=10)

def delete_last():
    txt = output_text.cget("text")
    output_text.config(text=txt[:-1])

def clear_text():
    output_text.config(text="")

Button(btn_frame, text="Delete", font=("Arial", 14), command=delete_last).grid(row=0, column=0, padx=10)
Button(btn_frame, text="Clear", font=("Arial", 14), command=clear_text).grid(row=0, column=1, padx=10)

# Guide
Label(right_frame, text="Guide Image", font=("Arial", 16, "bold"),
      bg="#222", fg="white").pack()

guide_canvas = Canvas(right_frame, width=200, height=200,
                      bg="#111", highlightthickness=1)
guide_canvas.pack(pady=5)

# Letter Buttons
letters = [
    ["A","B","C"], ["D","E","F"], ["G","H","I"],
    ["J","K","L"], ["M","N","O"], ["P","Q","R"],
    ["S","T","U"], ["V","W","X"], ["Y","Z","space"], ["del"]
]

btn_container = Frame(right_frame, bg="#222")
btn_container.pack(pady=10)

for r,row in enumerate(letters):
    for c,letter in enumerate(row):
        Button(btn_container, text=letter, width=10,
               command=lambda l=letter: update_guide_image(l)
               ).grid(row=r, column=c, padx=4, pady=4)

# ============================
# CAMERA LOOP WITH MEDIAPIPE
# ============================
cap = cv2.VideoCapture(0)

def update_camera():
    global stable_letter, frame_count, last_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_camera)
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_crop = None

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape

        # bounding box
        x_vals = []
        y_vals = []

        for lm in results.multi_hand_landmarks[0].landmark:
            x_vals.append(int(lm.x * w))
            y_vals.append(int(lm.y * h))

        x1, x2 = min(x_vals)-30, max(x_vals)+30
        y1, y2 = min(y_vals)-30, max(y_vals)+30

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        hand_crop = frame[y1:y2, x1:x2]

        # Prediction
        if hand_crop.size != 0:
            letter, conf = predict_letter(hand_crop)

            if letter == "nothing" or conf < CONF_THRESHOLD:
                prediction_label.config(text="Prediction: No hand")
                stable_letter, frame_count = "", 0

            else:
                prediction_label.config(text=f"Prediction: {letter} ({conf:.2f})")

                if letter == stable_letter:
                    frame_count += 1
                else:
                    stable_letter = letter
                    frame_count = 1

                if frame_count >= REQUIRED_FRAMES:
                    now = time.time()
                    if now - last_time > COOLDOWN_TIME:

                        txt = output_text.cget("text")

                        if letter == "space":
                            txt += " "
                        elif letter == "del":
                            txt = txt[:-1]
                        else:
                            txt += letter

                        output_text.config(text=txt)
                        update_guide_image(letter)

                        last_time = now
                        frame_count = 0
    else:
        prediction_label.config(text="Prediction: No hand")
        stable_letter = ""
        frame_count = 0

    # Show camera
    img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    camera_label.imgtk = img_tk
    camera_label.config(image=img_tk)

    root.after(10, update_camera)

update_camera()
root.mainloop()
