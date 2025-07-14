import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras import backend as K

# Config
IMG_SIZE = (128, 128)
DATA_DIR = "data/raw/test"
MODEL_PATH = "src/models/MobileNetV2_model.h5"
OUTPUT_DIR = "reports/figures/predictions"

# Setup
K.clear_session()
os.makedirs(OUTPUT_DIR, exist_ok=True)
model = load_model(MODEL_PATH, compile=False)

# Load test images
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='sparse',
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())
print("✅ Class names:", class_names)

# Loop through first 10 images
# Loop through first 10 images
for i in range(10):
    try:
        img_path = test_generator.filepaths[i]
        true_label = os.path.basename(os.path.dirname(img_path))
        orig_img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(orig_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        pred_index = np.argmax(prediction[0])
        pred_label = class_names[pred_index]
        confidence = prediction[0][pred_index] * 100

        # Prepare text
        text = f"True: {true_label} | Pred: {pred_label} ({confidence:.1f}%)"

        # Draw on image
        img_bgr = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
        cv2.putText(img_bgr, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Save
        out_path = os.path.join(OUTPUT_DIR, f"prediction_{i}_{pred_label}.jpg")
        cv2.imwrite(out_path, img_bgr)
        print(f"✅ Saved: {out_path}")

    except Exception as e:
        print(f"❌ Error on image {i}: {e}")
