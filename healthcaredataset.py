# -*- coding: utf-8 -*-
"""HealthcareDataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dvU_T3GAMR_4J60EKqzsjIIlOPqaEszs
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

healthcare = pd.read_csv('healthcare_dataset.csv')
print(healthcare.head())

len(healthcare) # 768 (this is the length of our file)
healthcare.dtypes # This is important for us to see the different data types of the variables
healthcare.describe() # This gives us a summary of the data

healthcare.isnull().sum() # No nulls

"""If there were nulls then we could have ran this code."""

#healthcare_clean = healthcare.dropna()
#len(healthcare_clean)
#healthcare_clean.isnull().sum # Rerunning to check for nulls

"""sns.pairplot(healthcare, hue = 'Outcome')
plt.show()
"""

sns.pairplot(healthcare, hue = 'Medical Condition')
plt.show()

# One-hot encoding for the 'category' feature
#encoder = OneHotEncoder(sparse = False)
#encoded_categories = encoder.fit_transform(diabetes[['category']])
#encoded_diabetes = pd.DataFrame(encoded_categories, columns = encoder.get_feature_names_out(['category']))

# Concatenate the encoded categories back to the original DataFrame
#df = pd.concat([diabetes.drop('category', axis=1), encoded_diabetes], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(healthcare.drop(columns=['Medical Condition']))
scaled_healthcare = pd.DataFrame(scaled_features, columns = healthcare.columns[:-1])
scaled_healthcare['Medical Condition'] = healthcare['Medical Condition']

# Dropping our target variable
X = healthcare.drop(columns=['Medical Condition'])
y = healthcare['Medical Condition']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# We can look at some of the values
#print(X_train)
print(X_train.describe()) # Descriptive statistics
#print(y_train)
#print(X_test)
#print(y_test)

"""models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Evaluate each model and store results
results = {}
for model_name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)
"""

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Evaluate each model and store results
results = {}
for model_name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)