
# Breast Cancer Classification (Imbalanced Dataset)

## Objective  
Classify breast cancer tumors as **malignant** or **benign** using multiple machine learning models.  
The focus was on handling **class imbalance** and improving **metrics** like precision, recall, F1-score, and ROC AUC.

---

## Dataset Overview

- **Source**: `sklearn.datasets.load_breast_cancer`
- **Samples**: 569
- **Features**: 30 numeric attributes (e.g., mean radius, texture, concavity)
- **Target**:
  - `0` = Malignant (cancerous)
  - `1` = Benign (non-cancerous)

---

## Problem Statement

Breast cancer detection is a binary classification problem with **slightly imbalanced data**.  
To address this, we applied oversampling techniques and tested multiple models.

---

## Imbalance Handling

- Applied **SMOTE (Synthetic Minority Oversampling Technique)** on the training set
- Used `class_weight='balanced'` for **Logistic Regression**
- Used `scale_pos_weight=1` in **XGBoost**

---

## Models Used

| Model                | Technique |
|---------------------|-----------|
| Logistic Regression | Balanced class weights + increased max_iter |
| Random Forest       | Balanced class weights + ensemble learning |
| XGBoost Classifier  | Boosted trees + class weight tuning |

---

## Evaluation Metrics

| Model               | Accuracy | Precision (malignant) | Recall (malignant) | ROC AUC |
|---------------------|----------|------------------------|---------------------|---------|
| Logistic Regression | 0.94     | 0.92                   | 0.93                | 0.97    |
| Random Forest       | 0.97     | 0.95                   | 0.96                | 0.98    |
| XGBoost             | 0.98     | 0.97                   | 0.96                | 0.99    |

*Note: Scores may vary slightly depending on the data split.*

---

## Evaluation Charts

- `class_distribution.png`
- `imbalanced_distribution.png`
- `smote_balanced_train.png`
- `confusion_matrix_logreg.png`, `confusion_matrix_rf.png`, `confusion_matrix_xgb.png`
- `roc_curve_all_models.png`

---

## Output Files

| File                               | Description                         |
|------------------------------------|-------------------------------------|
| `breast_cancer_classification.ipynb` | Complete code and workflow         |
| `best_breast_cancer_model.pkl`       | Serialized best model (XGBoost)    |
| `.png` charts                        | EDA + model evaluation visuals     |
| `README.md`                         | This documentation file            |

---

## Final Summary

```
Goal: Handling class imbalance, improve precision/recall/F1/ROC AUC.

Actions Taken:

| Step                          | Status                              |
|-------------------------------|-------------------------------------|
| Class imbalance handled       | SMOTE applied on training set       |
| Logistic Regression improved  | class_weight='balanced' + max_iter  |
| Multiple algorithms tested    | Logistic Regression, RF, XGBoost    |
| ROC Curve + F1 + Confusion    | Evaluated & Plotted                 |
| ROC AUC                       | 0.99 achieved with XGBoost          |

Result:
- Accuracy, Precision, Recall, F1-score all improved
- ROC AUC increased from ~0.97 to ~0.99

Here, I increased the metrics
```

---
