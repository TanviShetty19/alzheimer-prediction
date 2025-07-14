import matplotlib.pyplot as plt
import numpy as np
import os

# ========== METRICS ==========
models = ['Logistic Regression', 'Random Forest', 'SVM', 'Custom CNN', 'MobileNetV2', 'EfficientNetB0']

accuracy  =    [81.0, 100.0, 99.0, 65.8, 78.89, 43.8]
precision =    [77.0, 100.0, 99.0, None, 53.0, None]
recall    =    [81.0, 100.0, 99.0, None, 41.0, None]
f1_score  =    [78.0, 100.0, 99.0, None, 44.0, None]

# Output directory
output_dir = "reports/figures"
os.makedirs(output_dir, exist_ok=True)

# ========== 1. Accuracy Bar Chart ==========
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color='skyblue')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 110)

for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.1f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_comparison_chart.png"), dpi=300)
plt.close()

# ========== 2. Grouped Bar Chart ==========
x = np.arange(len(models))
width = 0.2

def safe_vals(vals):
    return [v if v is not None else 0 for v in vals]

plt.figure(figsize=(14, 6))
plt.bar(x - 1.5*width, safe_vals(accuracy), width, label='Accuracy', color='#4daf4a')
plt.bar(x - 0.5*width, safe_vals(f1_score), width, label='F1-Score', color='#377eb8')
plt.bar(x + 0.5*width, safe_vals(precision), width, label='Precision', color='#ff7f00')
plt.bar(x + 1.5*width, safe_vals(recall), width, label='Recall', color='#e41a1c')

plt.xticks(x, models, rotation=15)
plt.ylabel("Score (%)")
plt.title("ML vs DL Model Metric Comparison")
plt.ylim(0, 110)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grouped_model_comparison_chart.png"), dpi=300)
plt.close()

# ========== 3. Save Results as Text ==========
results = []
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i in range(len(models)):
    line = f"{models[i]}:"
    for m_name, m_vals in zip(metrics, [accuracy, precision, recall, f1_score]):
        val = m_vals[i]
        line += f" {m_name} = {val if val is not None else 'N/A'} |"
    results.append(line)

results_path = os.path.join(output_dir, "model_scores.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

# ========== 4. Determine Best Model (based on MobileNetV2) ==========

best_model_index = models.index('MobileNetV2')
best_model = models[best_model_index]

print(f"\n🏆 Best model (based on realistic generalization): {best_model} ({accuracy[best_model_index]:.2f}% Accuracy)")
print(f"📁 Results saved in: {output_dir}")
