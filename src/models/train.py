import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# ====================
# Reproducibility
# ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
K.clear_session()

# ====================
# GPU Check
# ====================
print("\u2705 TensorFlow GPU Available:", tf.config.list_physical_devices('GPU'))
try:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\u2705 GPU Available via torch: {device}")
except:
    print("\u26A0\uFE0F Torch not installed or GPU not available via torch.")

# ====================
# Config
# ====================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = "data/raw/images/"
BASE_MODEL_PATH = "src/models/"
PLOT_PATH = "reports/figures/best_model_accuracy.png"
os.makedirs(BASE_MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

# ====================
# Data Loaders
# ====================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='sparse', subset='training', shuffle=True, seed=SEED
)
val_generator = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='sparse', subset='validation', shuffle=False, seed=SEED
)

# ====================
# Class Weights
# ====================
labels = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

# ====================
# Model Variants
# ====================
def build_custom_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])
    return model

def build_mobilenet():
    base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                              include_top=False, weights='imagenet', pooling='avg')
    model = Sequential([
        base_model,
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])
    return model

def build_efficientnet():
    base_model = EfficientNetB0(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                                 include_top=False, weights='imagenet', pooling='avg')
    model = Sequential([
        base_model,
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])
    return model

# ====================
# Train and Evaluate
# ====================
results = []
models = {
    "CustomCNN": build_custom_cnn,
    "MobileNetV2": build_mobilenet,
    "EfficientNetB0": build_efficientnet
}

for name, builder in models.items():
    print(f"\n\U0001F680 Training {name}...")
    K.clear_session()
    model = builder()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path = os.path.join(BASE_MODEL_PATH, f"{name}_model.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )

    val_acc = max(history.history['val_accuracy'])
    results.append((name, val_acc, checkpoint_path, history))
    print(f"\u2705 {name} Best Validation Accuracy: {val_acc:.4f}")

# ====================
# Select Best Model
# ====================
best_model_name, best_acc, best_model_path, best_history = max(results, key=lambda x: x[1])
print(f"\n\U0001F3C6 Best Model: {best_model_name} with Validation Accuracy: {best_acc:.4f}")

# ====================
# Save Plot of Best
# ====================
plt.figure(figsize=(8, 5))
plt.plot(best_history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title(f"Accuracy - Best Model: {best_model_name}")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()

print("\u2705 Best Model saved to:", best_model_path)
print("\ud83d\udcc8 Plot saved to:", PLOT_PATH)
