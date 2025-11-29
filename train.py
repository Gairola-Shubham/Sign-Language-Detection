import json
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path

IMG_SIZE = 128
BATCH_SIZE = 32
DATA_DIR = "data"

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")

print("Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.08,
    brightness_range=(0.8, 1.2),
    shear_range=0.05,
    horizontal_flip=False,     # don't flip sign language
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_gen.num_classes
print("Classes found:", train_gen.class_indices)

print("Building model...")

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
out = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
]

print("Training first stage...")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=8,
    callbacks=callbacks
)

print("Fine-tuning...")

base.trainable = True
fine_tune_at = len(base.layers) - 50
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)

print("Training complete. Saving metadata...")

Path("artifacts").mkdir(exist_ok=True)

# save index-to-class mapping
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
with open("artifacts/class_indices.json", "w") as f:
    json.dump(idx_to_class, f)

# save preprocess config
with open("artifacts/preprocess.json", "w") as f:
    json.dump({"img_size": IMG_SIZE, "rescale": 1./255}, f)

print("Done. Best model saved as best_model.h5")
