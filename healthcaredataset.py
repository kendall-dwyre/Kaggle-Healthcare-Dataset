# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
healthcare = pd.read_csv('healthcare_dataset.csv')

# Preview the first few rows of the dataset
print(healthcare.head())

# Check the dataset info and missing values
print(f"Number of records: {len(healthcare)}")
print(healthcare.dtypes)
print(f"Missing values per column:\n{healthcare.isnull().sum()}")

# Exploratory Data Analysis (EDA) - Visualizing data distributions
sns.pairplot(healthcare, hue='Medical Condition')
plt.show()

# Separate numeric and categorical columns
numeric_columns = ['Age', 'Billing Amount', 'Room Number']
categorical_columns = ['Gender', 'Blood Type', 'Medical Condition', 'Doctor', 'Hospital', 'Insurance Provider', 'Admission Type', 'Medication', 'Test Results']

# Standardizing the Numeric Features
scaler = StandardScaler()
scaled_numeric = pd.DataFrame(scaler.fit_transform(healthcare[numeric_columns]), columns=numeric_columns)

# Encoding Categorical Features
encoder = OneHotEncoder(sparse_output=False)  # Changed from sparse to sparse_output
encoded_categorical = pd.DataFrame(encoder.fit_transform(healthcare[categorical_columns]), columns=encoder.get_feature_names_out(categorical_columns))

# Concatenate the scaled numeric and encoded categorical data
processed_healthcare = pd.concat([scaled_numeric, encoded_categorical], axis=1)

# Define features (X) and target (y)
X = processed_healthcare.drop(columns=encoder.get_feature_names_out(['Medical Condition']))
y = healthcare['Medical Condition']  # Target variable remains unencoded for now

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models to be evaluated
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

    # Model performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Using weighted average to handle multiple classes
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return accuracy, precision, recall, f1

# Evaluate each model and store the results
results = {}
for model_name, model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Display the results
results_df = pd.DataFrame(results).T
print("Model Performance Results:\n", results_df)

# Visualizing model performance
results_df.plot(kind='bar', figsize=(12, 8), title="Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()
