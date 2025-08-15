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


## 🧩 LIME Function in This Project

In this notebook, the **LIME** function is used to explain **why** the Random Forest model predicted a specific label for a given breast cancer sample.

* **`LimeTabularExplainer`** was initialized with:

  * Training features (`X_train`)
  * Feature names from the dataset
  * Class names: `"Malignant"` and `"Benign"`
  * `mode='classification'`

* **`explainer.explain_instance()`** was called on a **single test sample**, passing:

  * The feature values for that sample
  * The Random Forest model’s `.predict_proba` method
  * Number of features to show in the explanation

* The output is a **local interpretation** showing which features contributed most to the model’s prediction and whether they pushed toward `"Malignant"` or `"Benign"`.

Example:

> If `mean radius` was high and `smoothness mean` was above the average, LIME might show both as strong positive contributors to predicting `"Malignant"`.

---

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
