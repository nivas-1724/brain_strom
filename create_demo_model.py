"""
Demo Model Creator
Creates a lightweight untrained model just to test the full pipeline
(API + Grad-CAM + Web UI) before real training is complete.

Run: python create_demo_model.py
Then start the server: python backend/app.py

NOTE: This demo model gives RANDOM predictions.
      Train the real model with: python model/train.py
"""

import os
import sys
import json
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

SAVE_DIR   = "model/saved"
MODEL_NAME = "brain_tumor_model.h5"
NUM_CLASSES = 4

os.makedirs(SAVE_DIR, exist_ok=True)

print("🔧 Creating demo EfficientNetB0 model (untrained, for UI testing)...")

base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False

inputs  = keras.Input(shape=(224, 224, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(512, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model_path = os.path.join(SAVE_DIR, MODEL_NAME)
model.save(model_path)
print(f"✅ Demo model saved → {model_path}")

# Save class indices
class_indices = {
    "glioma":     0,
    "meningioma": 1,
    "notumor":    2,
    "pituitary":  3,
}
idx_path = os.path.join(SAVE_DIR, "class_indices.json")
with open(idx_path, "w") as f:
    json.dump(class_indices, f, indent=2)
print(f"✅ Class indices saved → {idx_path}")

# Quick inference test
dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
preds = model.predict(dummy, verbose=0)[0]
idx   = np.argmax(preds)
cls   = list(class_indices.keys())[idx]
print(f"\n✅ Test inference: predicted '{cls}' with {preds[idx]*100:.1f}% confidence")
print("\n⚠️  This is an UNTRAINED demo model — predictions are not meaningful.")
print("   Train the real model: python model/train.py\n")
print("🚀 Now start the server:  python backend/app.py")
