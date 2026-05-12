# Steel Plates Faults Classification: End-to-End ML Pipeline 🔩

This repository contains a comprehensive, professional Machine Learning pipeline for classifying surface defects in steel plates using the **UCI Steel Plates Faults** dataset.

## 📋 Project Overview
The goal of this project is to accurately predict 7 different types of steel surface defects by processing complex manufacturing data. The pipeline is designed to be modular, reproducible, and includes advanced techniques like feature selection voting and hyperparameter optimization.

## 🛠️ Pipeline Architecture
The project follows a structured data science workflow:

1.  **Exploratory Data Analysis (EDA):** Detailed statistical analysis, correlation studies using cosine similarity, and distribution checks.
2.  **Pre-processing:** * Missing value imputation using `KNNImputer`.
    * Outlier detection with `Local Outlier Factor (LOF)`.
    * Handling class imbalance via `SMOTE` (Synthetic Minority Over-sampling Technique).
3.  **Feature Engineering & Selection:** A robust voting-based selection mechanism comparing:
    * ANOVA F-test
    * Random Forest Feature Importance
    * Recursive Feature Elimination (RFE)
4.  **Modeling:** Comparative analysis of **LightGBM** across three data representations:
    * Full feature set
    * Selected "Core" features
    * PCA (Principal Component Analysis) reduced dimensions
5.  **Optimization:** Automated hyperparameter tuning using **Optuna**.
6.  **Evaluation & Explainability:**
    * Metrics: Accuracy, F1-Score, Precision, Recall, and AUC-ROC.
    * Visualizations: Confusion Matrices, ROC Curves, and Precision-Recall comparisons.
    * Interpretability: Model explainability using **SHAP (SHapley Additive exPlanations)**.

## 🚀 Technologies Used
* **Language:** Python 3.10+
* **Libraries:** Scikit-learn, LightGBM, Optuna, Pandas, Matplotlib, Seaborn, SHAP, Imbalanced-learn.
* **Environment:** Jupyter Notebook (.ipynb)

## 📊 Results Summary
The pipeline provides a comparative dashboard that evaluates the performance trade-offs between using the full dataset versus selected features, helping to identify the most efficient model for production.

---
*Created as part of an advanced machine learning study on industrial fault detection.*
