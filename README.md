# 🚢 Titanic Survival Prediction (Kaggle) — End-to-End ML Pipeline

**Dataset:** [Titanic - Machine Learning from Disaster – Kaggle](https://www.kaggle.com/c/titanic)

Predicting passenger survival on the Titanic using a complete machine-learning workflow:  
**data cleaning → feature engineering → model comparison → best-model selection → Kaggle submission**.

**Best validation performance:** **XGBoost = 0.8444 accuracy**

---

## ✨ Highlights
- Built a reproducible pipeline that merges train/test, engineers' features, compares models, and exports a Kaggle-ready submission.
- Feature engineering from raw fields (e.g., **Title from Name**, **Deck/Floor from Cabin**, **FamilySize**, **Single**, **Embarked dummies**).
- Benchmarked **6 classification models** and automatically selected the best one by validation accuracy.
- Final output file: `data/final/Titanic_Machine_Learning_from_Disaster.csv`.

---

## 🧠 Approach (What I did)

### 1) Data preprocessing
- Merged train + test with a flag (`is_train`) to ensure consistent transformations.
- Encoded variables (e.g., `Sex` mapped to numeric).
- Imputed missing values:
  - `Age` → median
  - `Fare` → median
  - `Embarked` → mode

### 2) Feature engineering (adds signal beyond baseline)
- **Title extraction** from passenger names + grouping rare titles.
- **Cabin deck (“Floor”)** extracted from Cabin and one-hot encoded (A–G).
- **Family features**
  - `FamilySize = SibSp + Parch + 1`
  - `Single` indicator
- One-hot encoding for Embarked (kept C and Q).

### 3) Modeling + evaluation
Models compared (same validation split, `random_state=42`):
- Logistic Regression (scaled features)
- SVC (scaled features)
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost

Validation accuracy summary (best model highlighted):
| Model | Accuracy (Valid) |
|---|---:|
| Logistic Regression | 0.8333 |
| SVC | 0.8333 |
| Random Forest | 0.8222 |
| Gradient Boosting | 0.8333 |
| AdaBoost | 0.8000 |
| **XGBoost** | **0.8444** |

---

## 📁 Repo structure
- `Titanic_Machine_Learning.py` — main pipeline (preprocessing → modeling → submission)
- `Titanic.ipynb` — notebook version (optional)
- `Exploratory_Data_Analysis_(EDA).ipynb` — EDA & visuals (optional)
- `data/train.csv`, `data/test.csv` — Kaggle dataset files (expected locally)
- `data/final/` — output submission file

---

## 🚀 How to run
> Note: the script expects dataset files under `data/` (see code paths).

### 1) Install dependencies
If you have a `requirements.txt`:
```bash
pip install -r requirements.txt
