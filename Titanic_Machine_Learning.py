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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
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
# 3. Data Preprocessing — structural feature engineering only.
# =====================================================
# NOTE: everything in this section is either a pure transform of an existing
# value (Sex mapping, Title/Floor extraction) or a categorical encoding whose
# columns must be identical across train/val/test (Title, Floor one-hot).
# None of it depends on a statistic (median/mode) computed across rows, so
# doing it on the combined train+test frame is safe and does not leak
# information across the eventual train/validation split.
titanic = pd.concat([train_df, test_df], axis=0)
 
## 3.1 Encode Gender into numerical values
# FIX: .replace() with parallel lists is the pattern pandas is moving away
# from (silent-downcasting warnings on recent 2.x). .map() is the explicit,
# future-proof equivalent for a hard categorical mapping.
titanic["Sex"] = titanic["Sex"].map({"male": 0, "female": 1})
 
## 3.2 Extract Title from Name column
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
### One-Hot Encode Floor feature
# FIX: previously selected a hardcoded ["Floor_A", ..., "Floor_G"] list,
# which silently drops the single "T" deck cabin and would raise a KeyError
# outright if any of A-G were ever absent from the data. reindex() with a
# fixed, explicit column set is robust to both: missing decks become an
# all-False column, unexpected/rare decks (like T) are still accounted for.
floor_dum = pd.get_dummies(titanic["Floor"], prefix="Floor", prefix_sep="_")
expected_decks = [f"Floor_{d}" for d in "ABCDEFGT"]
floor_dum = floor_dum.reindex(columns=expected_decks, fill_value=False)
titanic = pd.concat([titanic, floor_dum], axis=1)
 
## 3.4 Family Features
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"] + 1
titanic["Single"] = np.where(titanic["FamilySize"] == 1, 1, 0)
 
## 3.5 Split Back into Train and Test Sets
train_df = titanic[titanic["is_train"] == 1].drop(columns="is_train").reset_index(drop=True)
test_df = titanic[titanic["is_train"] == 0].drop(columns=["is_train", "Survived"]).reset_index(drop=True)
 
print(f"Shape of Training Dataset after preprocessing: {train_df.shape}")
print(f"Shape of Test Dataset after preprocessing: {test_df.shape}")
 
## 3.6 Drop Unnecessary Columns
# NOTE: "Embarked" is deliberately KEPT here (not filled, not one-hot-encoded
# yet) — see section 4.3, where its missing values and encoding are handled
# strictly after the train/validation split.
col_drop = ["Name", "Ticket", "Cabin", "Title", "Floor"]
train_df = train_df.drop(columns=col_drop)
test_df = test_df.drop(columns=col_drop)
 
# =====================================================
# 4. Model Training and Evaluation
# =====================================================
## 4.1 Define Features and Target
features = [c for c in train_df.columns if c not in ["PassengerId", "Survived"]]
X = train_df[features]
y = train_df["Survived"]
 
## 4.2 Train-Validation Split — done BEFORE any missing-value statistic
# (Age/Fare median, Embarked mode) is computed.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
 
## 4.3 Leakage-safe missing-value imputation + Embarked encoding
# FIX: Age/Fare/Embarked fill values used to be computed on the full
# train+test concatenation (`titanic`) before any split — meaning both the
# public Kaggle test set and this internal validation split contributed to
# the statistics used to fill training rows. Fit is now done on X_train
# ONLY, then applied to X_val and test_df — the same leakage-safe pattern
# used for the cluster price statistics in the London House Price project.
age_median = X_train["Age"].median()
fare_median = X_train["Fare"].median()
embarked_mode = X_train["Embarked"].mode()[0]
 
def clean_and_encode(df):
    df = df.copy()
    df["Age"] = df["Age"].fillna(age_median)
    df["Fare"] = df["Fare"].fillna(fare_median)
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    embarked_dum = pd.get_dummies(df["Embarked"], prefix="Embarked", prefix_sep="_")
    embarked_dum = embarked_dum.reindex(columns=["Embarked_C", "Embarked_Q"], fill_value=False)
    return pd.concat([df.drop(columns="Embarked"), embarked_dum], axis=1)
 
X_train = clean_and_encode(X_train)
X_val = clean_and_encode(X_val)
test_df = clean_and_encode(test_df)
 
# Keep the features list in sync with the Embarked_C / Embarked_Q columns
features = X_train.columns.tolist()
 
## 4.4 Imputation and Scaling for Logistic Regression and SVC
# (Kept as a defensive safety net — by this point there should be no
# remaining NaNs, since Age/Fare/Embarked were just handled above and
# Title/Floor were one-hot-encoded with no missing category left over.)
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
 
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train_imputed)
X_scaled_val = scaler.transform(X_val_imputed)
 
## 4.5 Define Models
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
 
## 4.6 Train and Evaluate Models
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
 
    # FIX: the original had a stray `\n"` after this line with no opening
    # quote — a hard SyntaxError that prevented the whole script from being
    # parsed at all. Both confusion matrices are now wrapped consistently
    # with an f-string, matching what the Train line already did.
    results[name] = {
        "Accuracy Train": accuracy_score(y_train, y_pred_train),
        "Confusion Matrix Train": f"{confusion_matrix(y_train, y_pred_train)}\n",
        "Accuracy Valid": accuracy_score(y_val, y_pred_val),
        "Confusion Matrix Valid": f"{confusion_matrix(y_val, y_pred_val)}\n"
    }
 
## 4.7 Display Model Results
for name, scores in results.items():
    print(f"Model: {name}")
    for metric, value in scores.items():
        if isinstance(value, float):
            print(f"{metric:<25}: {value:.4f}")
        else:
            print(f"{metric:<25}:\n{value}")
    print("-" * 45)
 
## 4.8 Select Best Model and Predict on Test Set
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
