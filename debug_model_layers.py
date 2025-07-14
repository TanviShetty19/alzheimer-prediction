import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your saved model
model = load_model("src/models/MobileNetV2_model.h5", compile=False)

print("\n✅ Top-level layers:")
for layer in model.layers:
    print(" -", layer.name)

# Find MobileNetV2 base
mobilenet_base = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name:
        mobilenet_base = layer
        break

if mobilenet_base is None:
    print("\n❌ MobileNetV2 base NOT found in your model.")
else:
    print(f"\n✅ MobileNetV2 base model: {mobilenet_base.name}")
    print("📋 All Conv2D layers inside it:")
    for l in mobilenet_base.layers:
        if isinstance(l, tf.keras.layers.Conv2D):
            print("   ↳", l.name)

    print("\n🔚 Deepest layer:", mobilenet_base.layers[-1].name)
