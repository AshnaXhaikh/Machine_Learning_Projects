---

# 🩺 Breast Cancer Prediction with LIME Explainability

This project explores **model explainability** using the **LIME** (Local Interpretable Model-Agnostic Explanations) technique on the **Breast Cancer dataset** from scikit-learn.

## 📌 Objective

The goal was to train machine learning models for predicting breast cancer diagnosis (benign or malignant) and then apply LIME to understand **how individual predictions are made**.

## 📊 Dataset

* **Source**: Scikit-learn's built-in `load_breast_cancer()` dataset
* **Features**: Various tumor measurements (mean radius, mean texture, etc.)
* **Target**: Binary classification – `0: Malignant`, `1: Benign`

## 🛠 Steps Performed

### 1️⃣ Model Training

* **Decision Tree Classifier**

  * Accuracy: **92%**
  * Interpretation: Decision Trees are inherently interpretable, so LIME was not necessary here.

* **Random Forest Classifier**

  * Accuracy: **96%**
  * Chosen for LIME analysis since it's a black-box model.

### 2️⃣ LIME Explainability

* Used `lime.lime_tabular.LimeTabularExplainer` to:

  * Select a single test sample
  * Explain the prediction by showing the **most influential features**
  * Visualize how each feature contributes positively or negatively to the prediction

## 📈 Results

* **Decision Tree**: Good performance, easy to interpret without additional tools.
* **Random Forest + LIME**: Higher accuracy and interpretable explanations for individual predictions, showing the model's reasoning in human-readable terms.

## 🚀 Tech Stack

* Python
* Scikit-learn
* LIME
* NumPy, Pandas
* Matplotlib / Seaborn

## 📂 How to Run

```bash
pip install -r requirements.txt
jupyter notebook lime_breast_cancer.ipynb
```

---
