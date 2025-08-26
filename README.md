# 🚢 Titanic: Machine Learning from Disaster – Survival Prediction

[![Kaggle](https://img.shields.io/badge/Kaggle-View%20Project-blue?logo=kaggle)](https://www.kaggle.com/code/alexandroskanakis/titanic-survived-classifier)
[![Python](https://img.shields.io/badge/Python-3.12-green?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange?logo=xgboost)]

---

## 📊 Project Overview

This project aims to predict the survival of **Titanic passengers** using machine learning algorithms on passenger data, including age, sex, fare, class, and family relations. It features extensive preprocessing, feature engineering, and evaluation of multiple models with automated best-model selection.
This project aims to predict the survival of **Titanic passengers** using machine learning algorithms on passenger data, including age, sex, fare, class, and family relations. It features extensive preprocessing, feature engineering, and evaluation of multiple models with automated best-model selection.

---

## 🔍 Motivation

The Titanic dataset is a classic ML challenge: predicting survival from structured data reflecting social, demographic, and economic factors. This project refines baseline efforts through advanced feature engineering (e.g., titles, family size, deck extraction) and compares tree-based and linear models to improve predictive accuracy.

---

## 📘 Dataset Overview

The dataset consists of:

<div align="center">

| File                                  | Description                                         |
|-------------------------------------|-----------------------------------------------------|
| `train.csv`                         | Training data with labeled survival outcome          |
| `test.csv`                          | Test data without survival labels to predict          |
| `data/final/`                      | Folder for storing model submission CSV files         |
| `notebooks/Exploratory_Data_Analysis_(EDA).ipynb` | Jupyter notebook containing exploratory data analysis |
| `requirements.txt`                 | Python package dependencies                            |

</div>

### ✨ Key Variables

<div align="center">

| Variable       | Description                                               |
|----------------|-----------------------------------------------------------|
| `PassengerId`  | Unique passenger identifier                                |
| `Survived`     | Survival status (0 = No, 1 = Yes)                         |
| `Pclass`       | Passenger class (1st, 2nd, 3rd)                           |
| `Sex`          | Gender of passenger                                       |
| `Age`          | Age in years                                             |
| `SibSp`        | Number of siblings/spouses aboard                         |
| `Parch`        | Number of parents/children aboard                         |
| `Ticket`       | Ticket number                                            |
| `Fare`         | Passenger fare                                          |
| `Cabin`        | Cabin number/floor (used for feature extraction)           |
| `Embarked`     | Port of embarkation (`C`=Cherbourg, `Q`=Queenstown, `S`=Southampton) |

</div>

---

## 🎯 Project Objective

Build a robust classification model to **predict survival** of Titanic passengers leveraging:

- Data cleaning and imputation of missing values  
- Feature engineering: titles, family size, cabin deck, etc.  
- Training and validation of multiple machine learning algorithms  
- Selecting the best model by validation accuracy  
- Generating submission files for the Kaggle competition

---

## 🏆 Achievements

- Created enriched features from raw data to improve predictive power  
- Compared and tuned six ML models, including Logistic Regression, SVC, Random Forest, Gradient Boosting, AdaBoost, and XGBoost  
- Achieved best validation accuracy with **XGBoost (84.44%)**  
- Automated pipeline for preprocessing, training, evaluation, and test prediction  
- Prepared final submission file ready for Kaggle upload  

---

## 🔧 Tools & Technologies

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Platform:** Kaggle and local Python environment  

---

## 📁 Repository Contents

<div align="center">

| File                                | Description                                           |
|-----------------------------------|-------------------------------------------------------|
| `Titanic_Machine_Learning.py`     | Full pipeline: preprocessing, modeling, evaluation, prediction |
| `train.csv`                       | Labeled training set                                  |
| `test.csv`                        | Test set for prediction                               |
| `data/final/Titanic_Machine_Learning_from_Disaster.csv` | Submission file with model predictions                |
| `notebooks/Exploratory_Data_Analysis_(EDA).ipynb` | Notebook for initial data analysis                    |
| `requirements.txt`                | Required Python packages                              |

</div>

---

## 📂 Project Directory Structure

Titanic-ML-Project/  
├── data/  
│ ├── final/  
│ │ └── Titanic_Machine_Learning_from_Disaster.csv  
│ ├── test.csv  
│ └── train.csv  
├── notebooks/  
│ └── Exploratory_Data_Analysis_(EDA).ipynb  
├── README.md  
├── requirements.txt  
└── Titanic_Machine_Learning.py  

- **data/**: Data files for training, testing, and output  
- **data/final/**: Folder to save submission CSV  
- **notebooks/**: Analysis notebook with visualizations  
- **requirements.txt**: Dependencies list  
- **Titanic_Machine_Learning.py**: Main executable script  

---

## 🚀 Project Workflow Diagram

A[Load Data] --> B[Preprocessing & Cleaning]  
B --> C[Feature Engineering]  
C --> D[Train/Validation Split]  
D --> E[Model Training]  
E --> F[Model Evaluation]  
F --> G[Best Model Selection]  
G --> H[Predict on Test Set]  
H --> I[Save Submission File]  
