# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Data Importing
train_df = pd.read_csv('C:/Users/alexa/OneDrive/Έγγραφα/Work_Python/Data_Analysis/Project 1 Titanic/train.csv')
test_df = pd.read_csv('C:/Users/alexa/OneDrive/Έγγραφα/Work_Python/Data_Analysis/Project 1 Titanic/test.csv')
gender_submission = pd.read_csv('C:/Users/alexa/OneDrive/Έγγραφα/Work_Python/Data_Analysis/Project 1 Titanic/gender_submission.csv')

# # Data Inspection
# def inspect_dataframe(df, name="Dataset"):
#     print(f"Head {name}:\n", df.head(), "\n")
#     print(f"Tail {name}:\n", df.tail(), "\n")
#     print(f"Shape of {name}:\n", {df.shape}, "\n")
#     print(f"Info of {name}:\n", {df.info()}, "\n")
#     print(f"Summary {name}:\n, {df.describe(include='all')}")
#     print(f"Missing values in {name}:\n", df.isna().sum(), "\n")

# inspect_dataframe(train_df, "Train dataset")
# inspect_dataframe(test_df, "Test dataset")
# inspect_dataframe(gender_submission, "Gender Submission dataset")

# # Exploratory Data Analysis  - EDA
# print('Distribution of the target variable:') # Check the people who survived
# print(train_df['Survived'].value_counts())
# print('_____________________________________________')
# print('Distribution of the target variable with respect to Pclass:') # Check the people who survived by Pclass
# print(train_df.groupby('Pclass')['Survived'].value_counts())
# print('_____________________________________________')
# print('Distribution of the target variable with respect to Embarked:') # Check the people who survived by Embarked
# print(train_df.groupby('Embarked')['Survived'].value_counts())
# train_df['Embarked'] = train_df['Embarked'].astype('category')
# print('_____________________________________________')
# print('Distribution of the target variable with respect to Sex:') # Check the people who survived by Sex
# print(train_df.groupby('Sex')['Survived'].value_counts())
# print('_____________________________________________')
# print('Distribution of the target variable with respect to SibSp:') # Check the people who survived by SibSp
# print(train_df.groupby('SibSp')['Survived'].value_counts())
# print('_____________________________________________')
# print('Distribution of the target variable with respect to Parch:') # Check the people who survived by Parch
# print(train_df.groupby('Parch')['Survived'].value_counts())
# print('_____________________________________________')
# print('Distribution of the target variable with respect to Age:') # Check the people who survived by Age
# print(train_df.groupby('Age')['Survived'].value_counts())
# print('_____________________________________________')
# print('Distribution of the target variable with respect to Fare:') # Check the people who survived by Fare
# print(train_df.groupby('Fare')['Survived'].value_counts())
# print('_____________________________________________')


# ## Exploratory Data Analysis Visualization - EDA Visualization
# # Check the people who survived
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# # Countplot for people who survived
# sns.countplot(x='Survived', data=train_df, palette=['red', 'blue'], ax=axes[0])
# axes[0].set_ylabel('People', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
# axes[0].set_xlabel('Survived', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
# axes[0].set_xticklabels(['Died', 'Survived'], fontsize=12, fontname='Times New Roman')
# axes[0].set_title('Countplot for Survived People', fontsize=16, fontweight='bold', fontname='Times New Roman')
# # Pie chart for Survived People
# train_df['Survived'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'blue'], labels=['Died', 'Survived'], ax=axes[1])
# axes[1].set_ylabel('')
# axes[1].set_title('Pie chart for Survived People', fontsize=16, fontweight='bold', fontname='Times New Roman')
# plt.tight_layout()
# plt.show()

# # Countplot for people who survived by Pclass
# fig, ax = plt.subplots(1, figsize=(12, 6))
# sns.countplot(x='Pclass', hue='Survived', data=train_df, palette=['red', 'blue'])
# ax.set_xlabel('Pclass', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
# ax.set_xticklabels(['1st class', '2nd class', '3rd class'], fontsize=12, fontname='Times New Roman')
# ax.set_title('Countplot for Survived People by Pclass', fontsize=16, fontweight='bold', fontname='Times New Roman')
# ax.legend(['Died', 'Survived'], title='People', loc='upper right')
# plt.tight_layout()
# plt.show()

# # Countplot for people who survived by Embarked
# fig, ax = plt.subplots(1, figsize=(12, 6))
# sns.countplot(x='Embarked', hue='Survived', data=train_df, palette=['red', 'blue'])
# ax.set_ylabel('People', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
# ax.set_xlabel('Embarked', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
# ax.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'], fontsize=12, fontname='Times New Roman')
# ax.set_title('Countplot for Survived People by Embarked', fontsize=16, fontweight='bold', fontname='Times New Roman')
# ax.legend(['Died', 'Survived'], title='People', loc='upper right')
# plt.tight_layout()
# plt.show()

# # Countplot for people who survived Sex
# fig, ax = plt.subplots(1, figsize=(12, 6))
# sns.countplot(x='Sex', hue='Survived', data=train_df, palette=['red', 'blue'])
# ax.set_ylabel('People', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
# ax.set_xlabel('Sex', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
# ax.set_xticklabels(['Male', 'Female'], fontsize=12, fontname='Times New Roman')
# ax.set_title('Countplot for Survived People by Sex', fontsize=16, fontweight='bold', fontname='Times New Roman')
# ax.legend(['Died', 'Survived'], title='People', loc='upper right')
# plt.tight_layout()
# plt.show()

# Data Preprocessing
## Fill in the missing values
train_df['Family'] = train_df['SibSp'] + train_df['Parch']
test_df['Family'] = test_df['SibSp'] + test_df['Parch']

## Create Features for Male & Female
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

## Create fetures for Embarked
train_df['Embarked'].fillna('S', inplace=True)
test_df['Embarked'].fillna('S', inplace=True)
train_df['Embarked'] = train_df['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})
test_df['Embarked'] = test_df['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

## Fill in the missing values for Age and Fare
age_median = train_df['Age'].median()
train_df['Age'].fillna(age_median, inplace = True)
test_df['Age'].fillna(age_median, inplace = True)

train_fare_median = train_df['Fare'].median()
test_df['Fare'].fillna(train_fare_median, inplace = True)

## Create fetures for Cabin
value_cabin1 = train_df['Cabin'].value_counts()
map_cabin1 = value_cabin1.to_dict()
train_df['Cabin'] = train_df['Cabin'].map(map_cabin1)
train_df['Cabin'].fillna(value=0, inplace=True)

value_cabin2 = test_df['Cabin'].value_counts()
map_cabin2 = value_cabin2.to_dict()
test_df['Cabin'] = test_df['Cabin'].map(map_cabin2)
test_df['Cabin'].fillna(value=0, inplace=True)

## Create fetures for Age Group
# Define age bins and corresponding labels
age_bins = [0, 1, 4, 12, 19, 39, 59, float('inf')]
age_labels = ['Infant', 'Toddler', 'Child', 'Teen', 'Adult', 'Middle Age Adult', 'Senior Adult']
# Apply the binning to both train_df and test_df
train_df['Age Group'] = pd.cut(train_df['Age'], bins=age_bins, labels=age_labels, right=False)
Age_Group_dummies = pd.get_dummies(train_df['Age Group'], prefix='Age')
train_df = pd.concat([train_df, Age_Group_dummies], axis=1)
train_df = train_df.drop('Age Group', axis=1)

test_df['Age Group'] = pd.cut(test_df['Age'], bins=age_bins, labels=age_labels, right=False)
Age_Group_dummies = pd.get_dummies(test_df['Age Group'], prefix='Age')
test_df = pd.concat([test_df, Age_Group_dummies], axis=1)
test_df = test_df.drop('Age Group', axis=1)


# print('Missing values in train dataset:', train_df.isna().sum()) # Check the missing values in the Training dataset
# print('_____________________________________________')
# print('Missing values in test dataset:', test_df.isna().sum()) # Check the missing values in the Testing dataset
# print('_____________________________________________')
# print('Columns in test dataset:', test_df.columns)

# train_df.drop(columns=['PassengerId', 'Survived']).hist()
# plt.show()

# test_df.drop(columns=['PassengerId']).hist()
# plt.show()
# ## Correlation Visualization
# correlation = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Age_Infant']].corr() # Compute the correlation matrix
# # Create a figure heatmap for Correlation
# fig, ax = plt.subplots(1, figsize=(16, 6))
# sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.3f', ax=ax)
# ax.set_title('Correlation between the variables', fontsize=16, fontweight='bold', fontname='Times New Roman')
# plt.tight_layout()
# plt.show()

# ## Covariance Visualization
# covariance = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']].cov() # Compute the covariance matrix
# # Create a figure heatmap for Covariance
# fig, ax = plt.subplots(1, figsize=(16, 6))
# sns.heatmap(covariance, annot=True, cmap='coolwarm', fmt='.3f', ax=ax)
# ax.set_title('Covariance between the variables', fontsize=16, fontweight='bold', fontname='Times New Roman')
# plt.tight_layout()
# plt.show()


# Create Machine Learning Model
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin'] # Features for model
X = train_df[features]
y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # Split the data into training and testing sets

# Define the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Random Forest': RandomForestClassifier()
}
# Train and evaluate each model
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_prob)
    }
# Print the results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print('_____________________________________________')
# Select the best model based on the evaluation metrics
best_model_name = max(results, key=lambda x: results[x]['ROC AUC'])
best_model = models[best_model_name]
best_model.fit(X_train, y_train)
print(f"Best model selected: {best_model_name}")


# Testing Model
test_df['Survived'] = best_model.predict(test_df[features]) # Make predictions on the test dataset
submission = test_df[['PassengerId','Survived']] # Create a DataFrame for submission
print(f1_score(submission['Survived'], gender_submission['Survived'])) # Check the accuracy of the model

## Create a figure barplot for Survived by Sex for Testing Dataset
# fig, ax = plt.subplots(figsize=(8, 6))
# sns.barplot(x='Sex', y='Survived', data=test_df, palette=['red', 'blue'])
# ax.set_ylabel('Survival Probability', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
# ax.set_xlabel('Sex', fontsize=14, color='black', fontweight='bold', fontname='Times New Roman')
# ax.set_xticks([0, 1])
# ax.set_xticklabels(['Male', 'Female'], fontsize=12, fontname='Times New Roman')
# ax.set_title('Classification Survival by Gender', fontsize=16, fontweight='bold', fontname='Times New Roman')
# plt.tight_layout()
# plt.show()


submission.to_csv('submission.csv', index=False) # Save the DataFrame to a CSV file
print("Submission file saved as 'submission.csv'")