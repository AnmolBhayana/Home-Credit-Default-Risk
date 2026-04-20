# 💳 Home Credit Default Risk — Loan Default Prediction

A machine learning project that predicts the likelihood of a loan applicant defaulting on their repayment, built using **Scikit-learn** on the real-world Home Credit dataset. The focus is on building a reliable, production-minded pipeline — not just a model.

---

## 📌 Overview

Financial institutions lose billions annually to loan defaults. This project builds a classification model that identifies high-risk applicants from demographic and financial data, helping lenders make smarter, data-driven decisions.

---

## 🎯 Features

- ✅ End-to-end ML pipeline — data cleaning → feature engineering → modelling → evaluation
- ✅ Handles severe class imbalance (far more non-defaulters than defaulters)
- ✅ Optimised for **recall** — minimising missed defaults is the priority
- ✅ Multiple models compared — Logistic Regression baseline vs Random Forest
- ✅ Cross-validation to prevent overfitting
- ✅ Feature importance analysis
- ✅ Clear visualisations of model performance

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3 |
| ML Models | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Dataset | Home Credit Default Risk (Kaggle) |

---

## 🏗️ Project Pipeline

```
Raw Data
   ↓
Exploratory Data Analysis (EDA)
   ↓
Data Cleaning & Missing Value Handling
   ↓
Feature Engineering & Selection
   ↓
Class Imbalance Handling
   ↓
Model Training (Logistic Regression → Random Forest)
   ↓
Cross-Validation & Hyperparameter Tuning
   ↓
Evaluation (Precision, Recall, F1, ROC-AUC)
```

---

## ⚙️ Approach

### Class Imbalance
The dataset is heavily imbalanced — most applicants don't default. To address this:
- Applied **class weight adjustment** in the model
- Evaluated using **recall and F1** rather than raw accuracy
- A missed default (false negative) is more costly than a false alarm

### Models Used
| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline — interpretable, fast |
| Random Forest | Final model — higher accuracy, captures non-linear patterns |

### Key Features
- Credit amount and annuity ratio
- Employment duration
- External credit scores
- Days since registration and ID change

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Run
```bash
git clone https://github.com/AnmolBhayana/Home-Credit-Default-Risk.git
cd Home-Credit-Default-Risk

# Download dataset from Kaggle and place in /data folder
# https://www.kaggle.com/competitions/home-credit-default-risk

python main.py
```

---

## 📊 Results

- Random Forest outperformed Logistic Regression baseline on recall and ROC-AUC
- Cross-validation confirmed model generalises well to unseen data
- Feature importance analysis revealed credit history and income ratios as strongest predictors

---

## 💡 Key Learnings

- Why accuracy is a misleading metric on imbalanced datasets
- How to think about the real-world cost of false negatives vs false positives
- The importance of cross-validation in avoiding overfitting
- Feature engineering impact on model performance

---

## 🔮 Future Improvements

- Add XGBoost and LightGBM for comparison
- Build a simple web interface for real-time risk scoring
- Add SHAP values for model explainability

---

## 👤 Author

**Anmol Bhayana**
[LinkedIn](https://linkedin.com/in/a-721a2) • [GitHub](https://github.com/AnmolBhayana)

---

> Dataset source: [Kaggle — Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
