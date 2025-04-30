# Titanic - Machine Learning from Disaster

### Data Source: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic)

## Kaggle Project
You can view my project and model code on Kaggle here:  
**[Titanic Survived Classifier](https://kaggle.com/code/alexandroskanakis/titanic-survived-classifier)**

## Overview

The Titanic dataset has been split into two groups:

- **Training Set (train.csv)**:  
  The training set is used to build machine learning models. It includes the outcome (ground truth) for each passenger, indicating whether they survived the sinking of the Titanic. The features provided in this dataset include attributes like passenger's gender, class, age, and more.

- **Test Set (test.csv)**:  
  The test set is used to evaluate your model's performance on unseen data. For this set, the ground truth (survival outcome) is not provided, and your task is to predict whether each passenger survived or not.

- **gender_submission.csv**:  
  A sample submission file that assumes all female passengers survived and male passengers did not. This is an example of how to structure a submission file for Kaggle.

## Data Dictionary

### Variable Definitions

- **PassengerId**: Unique ID for each passenger.
- **Survived**: 0 = No, 1 = Yes (Survival outcome).
- **Pclass**: Ticket class; 1 = 1st class, 2 = 2nd class, 3 = 3rd class.
- **Sex**: Gender of the passenger; Male, Female.
- **Age**: Age in years.
- **SibSp**: Number of siblings/spouses aboard the Titanic.
- **Parch**: Number of parents/children aboard the Titanic.
- **Ticket**: Ticket number.
- **Fare**: Passenger fare.
- **Cabin**: Cabin number.
- **Embarked**: Port of embarkation; C = Cherbourg, Q = Queenstown, S = Southampton.

### Variable Notes

- **Pclass**: Proxy for socio-economic status (SES).
  - 1st = Upper class
  - 2nd = Middle class
  - 3rd = Lower class

- **Age**: Age is fractional for infants (less than 1 year). If the age is estimated, it's in the form of xx.5.

- **SibSp**: Family relations defined as follows:
  - **Sibling** = brother, sister, stepbrother, stepsister
  - **Spouse** = husband, wife (mistresses and fiancés were excluded)

- **Parch**: Family relations defined as follows:
  - **Parent** = mother, father
  - **Child** = daughter, son, stepdaughter, stepson  
  Some children traveled only with a nanny, in which case **Parch** would be 0.

---

### **Goal**

The objective is to predict whether a passenger survived the Titanic disaster based on the provided features. By training a model on the training set and validating it on the test set, you can assess your model’s accuracy and submit predictions for the unseen passengers in the test set.

---

### **Future Work**

- Further **feature engineering** and model improvements.
- Exploring different **machine learning models** for better accuracy.
- Including **cross-validation** and **hyperparameter tuning**.

---

This version improves readability, uses consistent formatting for variable definitions, and adds clarity in the goals and future work sections. Let me know if you'd like additional information, or more sections added, such as project installation or execution instructions!
