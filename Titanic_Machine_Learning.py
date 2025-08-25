# =====================================================
# 1. Import Libraries
# =====================================================
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import os


# =====================================================
# 2. Data Importing
# =====================================================
train_df = pd.read_csv(r"data\train.csv")
test_df = pd.read_csv(r"data\test.csv")

# Add flag before merging datasets
train_df["is_train"] = 1
test_df["is_train"] = 0

print(f"Shape of Training Dataset: {train_df.shape}")
print(f"Shape of Test Dataset: {test_df.shape}")


# =====================================================
# 3. Data Preprocessing
# =====================================================
titanic = pd.concat([train_df, test_df], axis=0)


## 3.1 Data Cleaning
### Encode Gender into numerical values
titanic["Sex"] = titanic["Sex"].replace(["male", "female"], [0, 1])


## 3.2 Feature Engineering
### Extract Title from Name column
titanic["Title"] = titanic["Name"].str.extract(r" ([A-Za-z]+)\.")

### Replace rare titles with "Rare" category
titanic["Title"] = titanic["Title"].replace([
    "Capt", "Col", "Countess",
    "Don", "Dona", "Dr",
    "Jonkheer", "Lady", "Major",
    "Mlle", "Mme", "Ms",
    "Rev", "Sir"
], "Rare")

### One-Hot Encode the Title feature
title_dum = pd.get_dummies(titanic["Title"], drop_first=True)
titanic = pd.concat([titanic, title_dum], axis=1)


## 3.3 Cabin Feature Engineering
### Extract Floor/Deck information from Cabin column
titanic["Floor"] = titanic["Cabin"].str.extract(r"([A-Za-z]+)")

### One-Hot Encode Floor feature, selecting specific floors
floor_dum = pd.get_dummies(
    titanic["Floor"], prefix="Floor", prefix_sep="_"
)[[
    "Floor_A", "Floor_B", "Floor_C", "Floor_D",
    "Floor_E", "Floor_F", "Floor_G"
]]
titanic = pd.concat([titanic, floor_dum], axis=1)


## 3.4 Handling Missing Values
### Fill missing Fare values with median Fare
titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

### Fill missing Age values with median Age
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

### Fill missing Embarked values with mode Embarked
mode_Embarked = titanic["Embarked"].mode()[0]
titanic["Embarked"] = titanic["Embarked"].fillna(mode_Embarked)


## 3.5 Creating New Features
### Calculate Family Size
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"] + 1

### Create Single passenger indicator
titanic["Single"] = np.where(titanic["FamilySize"] == 1, 1, 0)

### One-Hot Encode Embarked feature, keep only Embarked_C and Embarked_Q
embarked_dummies = pd.get_dummies(
    titanic["Embarked"], prefix="Embarked", prefix_sep="_"
)[["Embarked_C", "Embarked_Q"]]
titanic = pd.concat([titanic, embarked_dummies], axis=1)


## 3.6 Split Back into Train and Test Sets
train_df = titanic[titanic["is_train"] == 1].drop(columns="is_train").reset_index(drop=True)
test_df = titanic[titanic["is_train"] == 0].drop(columns=["is_train", "Survived"]).reset_index(drop=True)

print(f"Shape of Training Dataset after preprocessing: {train_df.shape}")
print(f"Shape of Test Dataset after preprocessing: {test_df.shape}")


## 3.7 Drop Unnecessary Columns
col_drop = ["Name", "Ticket", "Cabin", "Embarked", "Title", "Floor"]
train_df = train_df.drop(columns=col_drop)
test_df = test_df.drop(columns=col_drop)


# =====================================================
# 4. Model Training and Evaluation
# =====================================================


## 4.1 Define Features and Target
features = train_df.drop(columns=["PassengerId", "Survived"]).columns.tolist()
X = train_df[features]
y = train_df["Survived"]


## 4.2 Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


## 4.3 Imputation and Scaling for Logistic Regression and SVC
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train_imputed)
X_scaled_val = scaler.transform(X_val_imputed)


## 4.4 Define Models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "SVC": SVC(
        C=0.6,
        cache_size=100,
        decision_function_shape="ovo",
        max_iter=1000,
        degree=1,
        gamma="auto",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=1,
        min_samples_split=6,
        n_estimators=600,
        oob_score=True,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        learning_rate=0.007,
        max_depth=6,
        max_features="sqrt",
        min_samples_leaf=4,
        min_samples_split=2,
        n_estimators=700,
        subsample=0.6,
        random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150,
        booster="gbtree",
        learning_rate=0.01,
        max_depth=7,
        min_child_weight=2,
        min_split_loss=0.4,
        subsample=1,
        tree_method="approx",
        random_state=42
    )
}


## 4.5 Train and Evaluate Models
results = {}

for name, model in models.items():
    # Use imputed+scaled features for Logistic Regression and SVC
    if name in ["Logistic Regression", "SVC"]:
        model.fit(X_scaled_train, y_train)
        y_pred_train = model.predict(X_scaled_train)
        y_pred_val = model.predict(X_scaled_val)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

    results[name] = {
        "Accuracy Train": f"{accuracy_score(y_train, y_pred_train)}",
        "Confusion Matrix Train": f"{confusion_matrix(y_train, y_pred_train)}\n",
        "Accuracy Valid": f"{accuracy_score(y_val, y_pred_val)}",
        "Confusion Matrix Valid": f"{confusion_matrix(y_val, y_pred_val)}\n"
    }


## 4.6 Display Model Results
for name, scores in results.items():
    print(f"Model: {name}")
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"{metric:<25}: {value:.4f}")
        else:
            print(f"{metric:<25}:\n{value}")
    print("-" * 45)


## 4.7 Select Best Model and Predict on Test Set
best_model_name = max(results, key=lambda x: results[x]["Accuracy Valid"])
best_model = models[best_model_name]
print(f"Best model selected: {best_model_name}")

# Impute and scale test features for Logistic Regression/SVC
test_imputed = imputer.transform(test_df[features])
test_scaled = scaler.transform(test_imputed)

# Predict using scaled features if model requires scaling, else use unscaled
if best_model_name in ["Logistic Regression", "SVC"]:
    test_df["Survived"] = best_model.predict(test_scaled)
else:
    test_df["Survived"] = best_model.predict(test_df[features])

# Prepare submission DataFrame
Titanic_submission = test_df[["PassengerId", "Survived"]]

# Save submission file
output_dir = r'data\final'
os.makedirs(output_dir, exist_ok=True)
Titanic_submission.to_csv(os.path.join(output_dir, "Titanic_Machine_Learning_from_Disaster.csv"), index=False)
print("Submission file saved as 'Titanic_Machine_Learning_from_Disaster.csv'")