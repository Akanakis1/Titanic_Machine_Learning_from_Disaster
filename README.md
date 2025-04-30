# 🚢 Titanic: Machine Learning from Disaster

### 📊 Data Source  
[Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)

### 🔗 Kaggle Project  
Check out my solution and code here:  
**[Titanic Survived Classifier – Alexandros Kanakis](https://kaggle.com/code/alexandroskanakis/titanic-survived-classifier)**

---

## 📁 Dataset Overview

The Titanic dataset is divided into two primary files:

- **`train.csv`**  
  Contains labeled data (ground truth) used to train machine learning models. It includes features like passenger class, age, gender, and survival status.

- **`test.csv`**  
  Includes similar features but without the survival outcome. The goal is to predict this outcome using your trained model.

- **`gender_submission.csv`**  
  A sample prediction file that assumes all female passengers survived and all males did not. This serves as a baseline for submission formatting.

---

## 📘 Data Dictionary

### ✨ Key Variables

| Variable       | Description |
|----------------|-------------|
| `PassengerId`  | Unique identifier for each passenger |
| `Survived`     | Survival status (0 = No, 1 = Yes) |
| `Pclass`       | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Sex`          | Gender of the passenger |
| `Age`          | Age in years (fractional if <1, estimated if xx.5) |
| `SibSp`        | Number of siblings/spouses aboard |
| `Parch`        | Number of parents/children aboard |
| `Ticket`       | Ticket number |
| `Fare`         | Passenger fare |
| `Cabin`        | Cabin number |
| `Embarked`     | Port of embarkation (`C` = Cherbourg, `Q` = Queenstown, `S` = Southampton) |

### 👨‍👩‍👧 Family-Related Features

- **SibSp**:  
  - Siblings = brother, sister, stepbrother, stepsister  
  - Spouses = husband, wife (mistresses and fiancés excluded)

- **Parch**:  
  - Parents = mother, father  
  - Children = daughter, son, stepdaughter, stepson  
  *(Some children traveling only with a nanny may have `Parch = 0`.)*

---

## 🎯 Project Objective

The goal is to **build a predictive model** that estimates the survival of Titanic passengers based on the available features. This includes:

- Data cleaning and preprocessing  
- Feature engineering  
- Model training and validation  
- Performance evaluation  
- Submission of predictions to Kaggle

---

## 🔧 Future Work & Enhancements

- 📌 Further feature engineering (e.g., titles, family size, cabin sections)  
- 🤖 Comparing various ML algorithms (e.g., Random Forest, SVM, Gradient Boosting)  
- 🛠️ Hyperparameter tuning and k-fold cross-validation  
- 📈 Use of advanced evaluation metrics (ROC-AUC, F1-Score)

---

## 🧠 Skills Demonstrated

- Data wrangling with **Pandas**
- Visual analysis with **Matplotlib** and **Seaborn**
- Machine learning with **Scikit-learn**
- Submission formatting for Kaggle competitions
- Iterative improvement through error analysis

---

## 🛠️ Installation (Optional)

```bash
git clone https://github.com/Akanakis1/Titanic_Machine_Learning_from_Disaster.git
cd Titanic_Machine_Learning_from_Disaster
pip install -r requirements.txt  # if provided
python titanic_classifier.py     # or run via Jupyter Notebook