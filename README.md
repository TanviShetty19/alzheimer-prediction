
```
# 🧠 Alzheimer's Disease Progression Prediction

This project aims to **analyze and predict Alzheimer's disease** using both **MRI imaging** and **clinical data** from the [OASIS dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images).

We implement a full **data science pipeline** with:
- 📊 Clinical EDA & preprocessing
- 🧪 ML & DL modeling
- 🎯 Metric-based comparison (accuracy, precision, recall, F1-score)
- 💻 Streamlit web app for user interaction with MRI predictions
- 📦 Final packaging for GitHub & Streamlit Cloud deployment

---

## 📁 Project Structure
```
alzheimer-prediction/
│
├── data/
│   ├── raw/
│   │   ├── clinical/
│   │   └── images/ (Mild Dementia, Moderate Dementia, etc.)
│   ├── processed/
│   │   ├── clinical/
│   │   ├── images/
│   │   └── merged/
│   └── test/
│       └── (sampled images for prediction)
│
├── notebooks/
│   ├── 1_eda.ipynb
│   ├── 2_preprocessing.ipynb
│   └── 3_modeling.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── accuracy_comparison_chart.png
│   │   ├── confusion_matrix.png
│   │   └── grouped_model_comparison_chart.png
│   └── predictions/
│       └── prediction_0.jpg to prediction_9.jpg
│
├── src/
│   ├── models/
│   │   ├── MobileNetV2_model.h5
│   │   ├── EfficientNetB0_model.h5
│   │   └── CustomCNN_model.h5
│   ├── pipelines/
│   ├── utils/
│   ├── train.py
│   ├── predict.py
│   └── visualize_model_comparison.py
│
├── app.py               # Streamlit app
├── README.md
└── requirements.txt
```

---

## 🧾 Dataset Used
We used **OASIS Alzheimer's dataset**, which includes:
- **MRI images** labeled under:
  - Mild Dementia
  - Moderate Dementia
  - Non Demented
  - Very Mild Dementia
- **Clinical CSVs** with features like:
  - MMSE, eTIV, SES, EDUC, Age, Group (diagnosis)
  - Patient visit longitudinal records

---

## 🔍 1. Clinical Data: EDA & Preprocessing
**EDA (1_eda.ipynb):**
- Grouped patient distribution
- Age and MMSE score histograms
- Boxplots: eTIV, nWBV across diagnosis
- Heatmap of correlations
- Z-score based outlier detection

**Preprocessing (2_preprocessing.ipynb):**
- Null handling (mean/mode imputation)
- Label encoding (Group, Gender, Hand)
- Min-Max normalization
- Saved splits (train/test IDs)

---

## 🧠 2. ML Modeling (Clinical Data)
Implemented in: `3_modeling.ipynb`

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 81.0%    | 77.0%     | 81.0%  | 78.0%    |
| Random Forest       | 100.0%   | 100.0%    | 100.0% | 100.0%   |
| Support Vector Machine | 99.0%| 99.0%     | 99.0%  | 99.0%    |

🔎 **Note**: ML models are suspected to be **overfitted** due to perfect scores on small test sets.

---

## 🧠 3. DL Modeling (MRI Images)
Implemented in: `src/train.py`

### ✅ Trained Models:
- `Custom CNN`
- `MobileNetV2` ✅ (best validation accuracy)
- `EfficientNetB0`

### ✅ Best Model: `MobileNetV2_model.h5`
- **Train Accuracy**: 85.0%
- **Validation Accuracy**: 76.2%
- **Test Accuracy**: 78.89%
- **Loss**: 0.5718

📋 **Classification Report:**
| Class                | Precision | Recall | F1-Score | Support |
|----------------------|-----------|--------|----------|---------|
| Mild Dementia        | 0.53      | 0.12   | 0.20     | 1250    |
| Moderate Dementia    | 0.36      | 0.41   | 0.39     | 122     |
| Non Demented         | 0.82      | 0.96   | 0.88     | 16805   |
| Very Mild Dementia   | 0.47      | 0.21   | 0.29     | 3431    |

🎯 **Weighted Avg F1**: 0.75  
📈 **Macro Avg F1**: 0.44

📌 Imbalance affects recall of rare classes. Further improvements can use:
- Focal loss
- Class weights
- Data oversampling

---

## 📊 4. Model Comparison (ML vs DL)
Executed in: `src/visualize_model_comparison.py`

![Accuracy Chart](reports/figures/accuracy_comparison_chart.png)  
![Grouped Metrics](reports/figures/grouped_model_comparison_chart.png)

🧠 **Best Generalization:** `MobileNetV2`  
🟢 Strong F1 on majority class  
🔴 Weak recall on minority dementia classes

---

## 🌐 5. Streamlit App (`app.py`)
**User Interface to:**
- Upload any MRI image
- Get prediction + confidence score
- Shows **predicted vs actual label** (if known)
- Log image result to `/reports/predictions/`

![Example Prediction](reports/predictions/prediction_1_Non_Demented.jpg)

---

## 🔄 6. How Predictions Work
Implemented in `src/predict.py`:
- Loads best DL model (`MobileNetV2_model.h5`)
- Samples real test images from each class
- Rescales & predicts
- Saves prediction + classification report + confusion matrix

All results saved in:
- `reports/figures/`
- `reports/predictions/`
- `classification_report.txt`

---

## 🧪 Grad-CAM (❌ Attempted, but skipped)
Although Grad-CAM was planned for visual explainability of CNN activations, it was **not used** due to layer naming mismatch and structural errors in final model.

Future versions can include explainability via Grad-CAM or SHAP on tabular data.

---

## 🚀 Deployment (Ready)
This project is:
- ✅ Tested Locally
- ✅ Packaged with `requirements.txt`
- ✅ Ready for GitHub push
- ✅ Ready to deploy on [Streamlit Cloud](https://streamlit.io/cloud)

---

## ✅ Requirements
```
Python 3.8+
tensorflow>=2.10
scikit-learn
matplotlib
seaborn
streamlit
pandas
numpy
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run
```bash
# 1. Activate conda or venv
conda activate env_alz

# 2. Run Streamlit app
streamlit run app.py

# 3. To predict from script
python src/predict.py

# 4. To view model comparison
python src/visualize_model_comparison.py
```

---

## 📌 Key Learnings
* DL handles unstructured MRI images better but requires balancing
* ML models were high performing but may be overfitted
* Streamlit made rapid deployment and UI testing easy
* Full pipeline execution from raw data to app was successful

---

## 🎯 Future Work
* Integrate Grad-CAM or XAI methods
* Use `TabNet` for clinical feature modeling
* Add CSV upload support for clinical input in UI
* Combine image + clinical input for multi-modal model

---

## 👤 Author
**Tanvi Shetty**  

Created with ❤️ and TensorFlow
```

