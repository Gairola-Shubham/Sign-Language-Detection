import cv2
import json
import argparse
import time
import numpy as np
from collections import deque
from pathlib import Path
import mediapipe as mp
import tensorflow as tf

# ---------------------------------------
# Parse arguments
# ---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="artifacts/best_model.h5")
parser.add_argument("--classes", type=str, default="artifacts/class_indices.json")
parser.add_argument("--preprocess", type=str, default="artifacts/preprocess.json")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--window", type=int, default=12)
parser.add_argument("--conf_thresh", type=float, default=0.7)
parser.add_argument("--cooldown", type=float, default=1.2)
parser.add_argument("--show_bbox", action="store_true")
args = parser.parse_args()

# ---------------------------------------
# Load model + metadata
# ---------------------------------------
model = tf.keras.models.load_model(args.model)

with open(args.classes, "r") as f:
    idx_map = json.load(f)
index_to_class = {int(k): v for k, v in idx_map.items()}

with open(args.preprocess, "r") as f:
    pp = json.load(f)

IMG_SIZE = int(pp.get("img_size", 128))
RESCALE = float(pp.get("rescale", 1/255))

# ---------------------------------------
# Mediapipe
# ---------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ---------------------------------------
# Camera Setup
# ---------------------------------------
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# TRUE FULLSCREEN WINDOW
cv2.namedWindow("ASL - Sentence Builder", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("ASL - Sentence Builder",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

# ---------------------------------------
# Variables
# ---------------------------------------
PAD = 40
smoothing = deque(maxlen=args.window)
sentence = ""
last_commit = 0
prev_time = time.time()
current_guide = "A"   # default guide image

def preprocess(crop):
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop = crop.astype("float32") * RESCALE
    return np.expand_dims(crop, 0)

print("Running FULLSCREEN...")
print("Controls:")
print("  A–Z = change guide")
print("  SPACE = space guide")
print("  X = del guide")
print("  d = delete sentence letter")
print("  c = clear all")
print("  q = quit")

# ---------------------------------------
# Main Loop
# ---------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to true fullscreen friendly resolution
    screen_w = 1920
    screen_h = 1080
    frame = cv2.resize(frame, (screen_w, screen_h))

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    display = "No hand"
    conf = 0

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]

        xs = [p.x for p in lm.landmark]
        ys = [p.y for p in lm.landmark]

        x_min = max(int(min(xs) * w) - PAD, 0)
        x_max = min(int(max(xs) * w) + PAD, w)
        y_min = max(int(min(ys) * h) - PAD, 0)
        y_max = min(int(max(ys) * h) + PAD, h)

        if x_max > x_min and y_max > y_min:
            crop = frame[y_min:y_max, x_min:x_max]
            img = preprocess(crop)
            pred = model.predict(img, verbose=0)[0]

            smoothing.append(pred)
            avg = np.mean(smoothing, axis=0)

            idx = int(np.argmax(avg))
            conf = float(avg[idx])
            label = index_to_class.get(idx, "UNK")

            if conf >= args.conf_thresh:
                display = f"{label} ({conf:.2f})"
            else:
                display = f"UNCERTAIN ({label}) {conf:.2f}"

            # Sentence commit logic
            now = time.time()
            if conf >= args.conf_thresh and (now - last_commit) > args.cooldown:

                if label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    sentence += label
                    last_commit = now

                elif label == "space":
                    sentence += " "
                    last_commit = now

                elif label == "del" and len(sentence) > 0:
                    sentence = sentence[:-1]
                    last_commit = now

        if args.show_bbox:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
    else:
        smoothing.clear()

    # FPS
    now = time.time()
    fps = 1 / (now - prev_time + 1e-9)
    prev_time = now

    # Prediction text
    cv2.putText(frame, display, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (0,255,0) if conf >= args.conf_thresh else (0,0,255), 3)

    # Sentence bar
    cv2.rectangle(frame, (10, 70), (w-10, 150), (0,0,0), -1)
    cv2.putText(frame, sentence, (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)

    # FPS display
    cv2.putText(frame, f"FPS: {int(fps)}", (20, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3)

    # ---------------------------------------
    # POSE GUIDE OVERLAY (top-right)
    # ---------------------------------------
    guide_path = f"guide/{current_guide}.jpg"
    if Path(guide_path).exists():
        guide_img = cv2.imread(guide_path)
        guide_img = cv2.resize(guide_img, (300, 300))

        gh, gw = guide_img.shape[:2]
        frame[20:20+gh, w-20-gw:w-20] = guide_img

    # ---------------------------------------
    # Controls
    # ---------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("d") and len(sentence) > 0:
        sentence = sentence[:-1]

    if key == ord("c"):
        sentence = ""

    if ord('a') <= key <= ord('z'):
        current_guide = chr(key).upper()

    if key == ord(" "):
        current_guide = "space"

    if key == ord("x"):
        current_guide = "del"

    if key == ord("n"):
        current_guide = "nothing"

    cv2.imshow("ASL - Sentence Builder", frame)

cap.release()
cv2.destroyAllWindows()
