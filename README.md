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
 
> **Note on interpreting the table below:** the six models are separated by
> only 4.4 points of accuracy (0.8000–0.8444) on a single held-out split of a
> ~891-row training set. A gap that small is well within what a different
> `random_state` could produce on its own. Treat XGBoost's win here as
> "roughly tied for best" rather than a decisive result unless it's confirmed
> with cross-validation (e.g. mean ± std accuracy across 5 folds) — that
> would also make the "disciplined evaluation" framing above fully earned
> rather than asserted.
 
---
 
## Feature Engineering Highlights
 
The following features were engineered to capture meaningful passenger patterns:
 
- **Title extraction** from names (with rare titles grouped)
- **Cabin deck (floor)** extracted from cabin identifiers
  <!-- TODO: state how missing Cabin values were handled before deck
  extraction (~77% of Cabin is missing in the raw Titanic data) — e.g.
  grouped as an "Unknown" deck, consistent with the categorical-imputation
  approach used elsewhere in this pipeline. -->
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
 
Models were trained and evaluated on the same held-out validation split
<!-- TODO: state the split ratio, e.g. "(80/20, random_state=42)" -->
(`random_state=42`) to ensure comparability.
 
| Model                      | Validation Accuracy |
|-----------------------------|---------------------|
| AdaBoost                    | 0.8000              |
| Random Forest                | 0.8222              |
| Logistic Regression          | 0.8333              |
| Support Vector Classifier    | 0.8333              |
| Gradient Boosting            | 0.8333              |
| **XGBoost**                  | **0.8444**          |
 
The best-performing model by validation accuracy was selected automatically
and used to generate the final submission file.
 
---
 
## Repository Structure
 
```
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── final/
│       └── Titanic_Machine_Learning_from_Disaster.csv
├── notebooks/
│   ├── Exploratory_Data_Analysis_(EDA).ipynb
│   └── Titanic.ipynb
├── Titanic_Machine_Learning.py
├── requirements.txt
└── README.md
```
 
---
 
## How to Run
 
1. Clone the repository:
```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
```
 
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
 
3. Run the pipeline:
```bash
   python Titanic_Machine_Learning.py
```
