# 🚢 Titanic: Machine Learning from Disaster – Survival Prediction

**Dataset:** [Titanic - Machine Learning from Disaster – Kaggle](https://www.kaggle.com/c/titanic)

---

## 📊 Project Overview

This project tackles the classic **Titanic survival prediction** challenge using machine learning algorithms on passenger data (age, sex, fare, class, etc.). The solution features a comprehensive pipeline involving **extensive preprocessing**, **advanced feature engineering**, and a comparative evaluation of **six different classification models** with automated best-model selection.

---

## 🔍 Motivation

The Titanic dataset serves as a benchmark for predicting survival from structured data reflecting social and economic factors. This project aims to surpass baseline results by applying **advanced feature engineering** (e.g., titles, family size, deck extraction) and rigorously comparing the performance of tree-based and linear models to achieve high predictive accuracy.

---

## 📘 Dataset Overview

The dataset consists of two primary files:

<div align="center">

| File | Description |
| :--- | :--- |
| `train.csv` | Training data with labeled survival outcome |
| `test.csv` | Test data without survival labels for prediction |
| `data/final/` | Folder for storing model submission CSV files |
| `notebooks/Exploratory_Data_Analysis_(EDA).ipynb` | Jupyter notebook containing exploratory data analysis |
| `requirements.txt` | Python package dependencies |

</div>

### ✨ Key Variables

<div align="center">

| Variable | Description |
| :--- | :--- |
| `Survived` | Survival status (**0 = No, 1 = Yes**) |
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Sex` | Gender of passenger |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number/floor (used for feature extraction) |
| `Embarked` | Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton) |

</div>

---

## 🎯 Project Objective

Build a robust classification model to **predict survival** of Titanic passengers, specifically by leveraging:

* Data cleaning and **imputation** of missing values.
* **Feature Engineering**: Extraction of **titles**, calculation of **family size**, and derivation of **cabin deck**.
* Training and validation of multiple machine learning algorithms.
* Selecting the best model based on validation accuracy.
* Generating submission files for the Kaggle competition.

---

## 🏆 Achievements

* Created **enriched features** from raw data to significantly improve predictive power.
* Compared and tuned **six ML models**: Logistic Regression, SVC, Random Forest, Gradient Boosting, AdaBoost, and XGBoost.
* Achieved best validation accuracy with **XGBoost (84.44%)**.
* Automated pipeline for preprocessing, training, evaluation, and test prediction.
* Prepared final submission file ready for Kaggle upload.

## 📊 Model Evaluation Results

The models were evaluated on the processed data using a train-validation split (Train N=712, Validation N=179). **XGBoost** was selected as the final model based on its superior performance:

<div align="center">

| Model Name | Accuracy Train | Accuracy Valid |
| :--- | :--- | :--- |
| **Logistic Regression** | 0.8377 | 0.8333 |
| **SVC** | 0.8502 | 0.8333 |
| **Random Forest** | 0.8826 | 0.8222 |
| **Gradient Boosting** | 0.8964 | 0.8333 |
| **AdaBoost** | 0.8140 | 0.8000 |
| **XGBoost** | **0.8814** | **0.8444** |

</div>

**Best Model Selected:** **XGBoost** (Validation Accuracy: 0.8444).
A submission file (`Titanic_Machine_Learning_from_Disaster.csv`) was generated using the best model, trained on the full dataset.

---

## 🔧 Tools & Technologies

* **Programming Language:** Python
* **Libraries:** **Pandas**, **NumPy**, **Scikit-learn**, **XGBoost**, Matplotlib, Seaborn
* **Platform:** Kaggle and local Python environment

---

## 📁 Repository Contents

<div align="center">

| File | Description |
| :--- | :--- |
| `Titanic_Machine_Learning.py` | Full pipeline: preprocessing, modeling, evaluation, prediction (Main executable script) |
| `train.csv`, `test.csv` | Dataset files |
| `data/final/Titanic_Machine_Learning_from_Disaster.csv` | Submission file with model predictions |
| `notebooks/Exploratory_Data_Analysis_(EDA).ipynb` | Notebook for initial data analysis and visualizations |
| `requirements.txt` | Required Python packages |

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

---

## 🚀 Project Workflow Diagram

A[Load Data] $\rightarrow$ B[Preprocessing & Cleaning] $\rightarrow$ C[Feature Engineering] $\rightarrow$ D[Train/Validation Split] $\rightarrow$ E[Model Training] $\rightarrow$ F[Model Evaluation] $\rightarrow$ G[Best Model Selection] $\rightarrow$ H[Predict on Test Set] $\rightarrow$ I[Save Submission File]

---
