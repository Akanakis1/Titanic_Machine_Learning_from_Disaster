# Titanic Survival Prediction — Reproducible Classification Workflow

This project implements a clean, end-to-end supervised classification workflow
using the Titanic passenger dataset. The focus is on **feature engineering,
model comparison, and disciplined evaluation**, rather than competition ranking.

Dataset source: https://www.kaggle.com/c/titanic

---

## Project Overview

**Objective**  
Predict passenger survival using structured demographic and ticket information,
and evaluate whether engineered features improve classification performance
over simple baselines.

**Workflow**  
Data cleaning → feature engineering → model comparison →
best-model selection → submission export.

---

## Key Result

- **Best validation accuracy:** **0.8444** (XGBoost)

This performance was achieved using engineered features derived from passenger
names, cabin information, and family structure.

---

## Feature Engineering Highlights

The following features were engineered to capture meaningful passenger patterns:

- **Title extraction** from names (with rare titles grouped)
- **Cabin deck (floor)** extracted from cabin identifiers
- **Family features**
  - `FamilySize = SibSp + Parch + 1`
  - `Single` indicator
- Encoded embarkation ports (C, Q)
- Consistent handling of missing values:
  - Age → median
  - Fare → median
  - Embarked → mode

---

## Modeling & Evaluation

Models were trained and evaluated using the same validation split
(`random_state=42`) to ensure comparability.

| Model               | Validation Accuracy |
|---------------------|---------------------|
| Logistic Regression | 0.8333              |
| Support Vector Classifier | 0.8333      |
| Random Forest       | 0.8222              |
| Gradient Boosting   | 0.8333              |
| AdaBoost            | 0.8000              |
| **XGBoost**         | **0.8444**          |

The best-performing model was selected automatically and used
to generate the final submission file.

---

## Repository Structure

├── data/  
│ ├── train.csv  
│ ├── test.csv  
│ └── final/  
│ └── Titanic_Machine_Learning_from_Disaster.csv  
├── notebooks/  
│ ├── Exploratory_Data_Analysis_(EDA).ipynb  
│ └── Titanic.ipynb  
├── Titanic_Machine_Learning.py  
├── requirements.txt  
└── README.md  

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
