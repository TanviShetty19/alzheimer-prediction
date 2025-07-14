import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==========================
# Configuration
# ==========================
SEED = 42
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DATA_DIR = "data/raw/images"
TEST_DIR = "data/raw/test"
BEST_MODEL_NAME = "MobileNetV2_model.h5"  # This should match best model from train.py
BEST_MODEL_PATH = f"src/models/{BEST_MODEL_NAME}"

# ==========================
# Step 1: Dynamic Test Set Creation
# ==========================
from sklearn.model_selection import train_test_split

TEST_SPLIT = 0.25
print("🔄 Dynamically creating test set by splitting each class...")

if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR, exist_ok=True)

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    random.seed(SEED)
    random.shuffle(images)

    test_count = max(5, int(len(images) * TEST_SPLIT))  # At least 5 images per class
    test_images = images[:test_count]

    dest_class_path = os.path.join(TEST_DIR, class_name)
    os.makedirs(dest_class_path, exist_ok=True)

    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(dest_class_path, img)
        shutil.copy(src, dst)

print("✅ Test set recreated with dynamic sampling.")

# ==========================
# Optional: Class Balance Check
# ==========================
for class_name in os.listdir(TEST_DIR):
    class_path = os.path.join(TEST_DIR, class_name)
    num_images = len([f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
    print(f"📂 {class_name}: {num_images} test samples")

# ==========================
# Step 2: Load Model
# ==========================
print("📥 Loading model from:", BEST_MODEL_PATH)
K.clear_session()
model = load_model(BEST_MODEL_PATH, compile=False)
model.compile(loss='sparse_categorical_crossentropy', metrics=[SparseCategoricalAccuracy()])

# ==========================
# Step 3: Load Test Data
# ==========================
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# ==========================
# Step 4: Evaluate & Predict
# ==========================
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

# ==========================
# Step 5: Visualize Predictions
# ==========================
class_names = list(test_generator.class_indices.keys())
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_labels = test_generator.classes

plt.figure(figsize=(12, 6))
for i in range(10):
    img, _ = test_generator.next()
    plt.subplot(2, 5, i + 1)
    plt.imshow(img[0])
    plt.axis('off')
    pred_label = class_names[predicted_classes[i]]
    true_label = class_names[int(true_labels[i])]
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=9)

plt.tight_layout()
plt.savefig("reports/figures/predicted_images.png")
plt.show()

# ==========================
# Step 6: Classification Report & Confusion Matrix
# ==========================
print("\n📋 Classification Report:\n")
print(classification_report(true_labels, predicted_classes, target_names=class_names))

cm = confusion_matrix(true_labels, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("reports/figures/confusion_matrix.png")
plt.show()
