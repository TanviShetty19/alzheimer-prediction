import matplotlib.pyplot as plt
import numpy as np

# === MODEL METRICS (You can update these with actual values) ===
models = ["Custom CNN", "MobileNetV2", "EfficientNetB0"]
accuracy = [0.712, 0.762, 0.789]
precision = [0.69, 0.74, 0.77]
recall = [0.71, 0.75, 0.79]
f1_score = [0.70, 0.75, 0.78]

# === VISUALIZATION ===
x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - 1.5*width, accuracy, width, label="Accuracy")
plt.bar(x - 0.5*width, precision, width, label="Precision")
plt.bar(x + 0.5*width, recall, width, label="Recall")
plt.bar(x + 1.5*width, f1_score, width, label="F1 Score")

plt.ylabel('Score')
plt.title('Model Comparison on Alzheimer\'s Dataset')
plt.xticks(x, models)
plt.ylim(0.5, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight best model
best_model_index = np.argmax(f1_score)
plt.text(best_model_index, f1_score[best_model_index] + 0.01, '🏆 Best', ha='center', fontsize=12, color='darkgreen')

plt.tight_layout()
plt.savefig("reports/figures/model_comparison_chart.png")
plt.show()
