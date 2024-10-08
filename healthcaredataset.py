# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset
healthcare = pd.read_csv('healthcare_dataset.csv')

# Previewing the first few rows of the dataset
print(healthcare.head())

# Data Summary and Info
print(f"Number of records: {len(healthcare)}")
print(healthcare.dtypes) # Check the data types of the columns
print(healthcare.describe()) # Statistical summary of the data
print(f"Missing values per column:\n{healthcare.isnull().sum()}") # Checking for missing values

# Exploratory Data Analysis (EDA) - Visualizing data distributions
sns.pairplot(healthcare, hue='Medical Condition')
plt.show()

# Standardizing the Features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(healthcare.drop(columns=['Medical Condition']))

# Creating a new DataFrame with scaled features
scaled_healthcare = pd.DataFrame(scaled_features, columns=healthcare.columns[:-1])
scaled_healthcare['Medical Condition'] = healthcare['Medical Condition']

# Splitting the dataset into features (X) and target (y)
X = scaled_healthcare.drop(columns=['Medical Condition'])
y = scaled_healthcare['Medical Condition']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display summary statistics for the training data
print(X_train.describe())

# Defining the models to be evaluated
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

# Displaying the results
results_df = pd.DataFrame(results).T
print("Model Performance Results:\n", results_df)

# Visualizing model performance
results_df.plot(kind='bar', figsize=(12, 8), title="Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()
