# Titanic Survival Prediction: Reproducible Classification Workflow
 
This project implements a clean, end-to-end supervised classification workflow using the Titanic passenger dataset. The focus is on **feature engineering, model comparison, and disciplined evaluation**, rather than competition ranking.
 
Dataset source: https://www.kaggle.com/c/titanic
 
---
 
## Project Overview
 
**Objective**
Predict passenger survival using structured demographic and ticket information, and evaluate whether engineered features improve classification performance over simple baselines.
 
**Workflow**
Data cleaning → feature engineering → model comparison → best-model selection → submission export.
 
---
 
## Key Result
 
- **Best cross-validated accuracy:** **0.841 ± 0.017** (XGBoost, stratified 5-fold)
- **Best single-split validation accuracy:** **0.8444** (XGBoost, 90 held-out passengers)

The features come from passenger names (title), cabin information (deck) and family structure.

> **How to read these numbers:** the validation split holds only 90 passengers, so
> one passenger is about 1.1 points of accuracy. On the 5-fold cross-validation, the
> top models (XGBoost 0.841, Gradient Boosting 0.837) are separated by less than
> one standard deviation (about 0.017). XGBoost is **roughly tied for best**, not a
> decisive winner. Models are therefore selected by the cross-validated mean.
 
---
 
## Feature Engineering Highlights
 
The following features were engineered to capture meaningful passenger patterns:
 
- **Title extraction** from names (with rare titles grouped)
- **Cabin deck (floor)** extracted from cabin identifiers (decks A–G); a
  missing Cabin value carries no deck signal at all (all deck indicators are
  0 for that passenger) rather than being grouped into an explicit "Unknown"
  category
- **Family features**
  - `FamilySize = SibSp + Parch + 1`
  - `Single` indicator
- One-hot encoded `Embarked` (Cherbourg, Queenstown; Southampton is the
  dropped reference category)
- Consistent handling of missing values:
  - Age → median
  - Fare → median
  - Embarked → mode
---
 
## Modeling & Evaluation
 
All models use the same 90/10 split (`random_state=42`) and the same stratified 5-fold
cross-validation (`random_state=42`). In each fold, the imputation values (Age and Fare median,
Embarked mode) and the scaler are learned from that fold's training portion only, so nothing
leaks from the held-out passengers.
 
| Model                      | Validation accuracy (90 passengers) | 5-fold CV accuracy (mean ± std) |
|-----------------------------|------------------------------------|---------------------------------|
| AdaBoost                    | 0.8000                             | 0.813 ± 0.016                   |
| Gradient Boosting           | 0.8222                             | 0.837 ± 0.020                   |
| Random Forest               | 0.8333                             | 0.823 ± 0.017                   |
| Logistic Regression         | 0.8333                             | 0.825 ± 0.019                   |
| Support Vector Classifier   | 0.8333                             | 0.828 ± 0.012                   |
| **XGBoost**                 | **0.8444**                         | **0.841 ± 0.017**               |
 
The model with the highest cross-validated mean is refit on the whole training set and used to generate
the submission file. Numbers come from scikit-learn 1.9.1 and xgboost 3.4.1; other library versions can
shift individual single-split values slightly.
 
---
 
## Repository Structure
 
```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── final/
│       └── Titanic_Machine_Learning_from_Disaster.csv
├── notebooks/
│   └── Exploratory_Data_Analysis_(EDA).ipynb
├── Titanic.ipynb
├── Titanic_Machine_Learning.py
├── requirements.txt
└── README.md
```
 
---
 
## How to Run
 
1. Clone the repository:
```bash
   git clone https://github.com/Akanakis1/Titanic_Machine_Learning_from_Disaster.git
   cd Titanic_Machine_Learning_from_Disaster
```
 
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
 
3. Run the pipeline:
```bash
   python Titanic_Machine_Learning.py
```
