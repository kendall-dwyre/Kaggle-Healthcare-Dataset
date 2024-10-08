# Healthcare Dataset - Machine Learning Project

## Overview

This project demonstrates machine learning techniques applied to a simulated healthcare dataset obtained from [Kaggle](https://www.kaggle.com/datasets/prasad22/healthcare-dataset). The dataset was created to mimic real-world healthcare data, providing a practical and educational platform for experimenting with healthcare analytics without compromising patient privacy. The goal of this project is to showcase my machine learning skills through data cleaning, exploratory analysis, feature engineering, model building, and evaluation.

The inspiration behind this dataset is rooted in the need for practical and diverse healthcare data for educational and research purposes. As the dataset creator explains:

> "Healthcare data is often sensitive and subject to privacy regulations, making it challenging to access for learning and experimentation. To address this gap, I have leveraged Python's Faker library to generate a dataset that mirrors the structure and attributes commonly found in healthcare records. By providing this synthetic data, I hope to foster innovation, learning, and knowledge sharing in the healthcare analytics domain."

## Dataset Description

The dataset simulates healthcare records and contains information on various patient attributes, including age, gender, blood type, and health-related metrics such as BMI, blood pressure, and medical conditions. It also includes details such as hospital admission type, insurance provider, billing amount, and medications.

### Key Features:
- **Age:** The patient's age.
- **Gender:** Male or female.
- **BMI:** Body Mass Index (weight in kg / height in m²).
- **Blood Type:** The patient's blood type.
- **Medical Condition:** Categorical target variable that identifies the patient's diagnosis.
- **Billing Amount:** The hospital billing information.
- **Insurance Provider:** The patient's insurance provider.
- **Hospital Admission Details:** Information about the patient's admission and discharge.

This data allows for classification tasks, predictive modeling, and exploratory data analysis (EDA), making it an excellent resource for practicing and refining healthcare analytics skills.

## Project Workflow

This project follows a typical data science workflow, outlined below:

### 1. **Define the Problem**
   Although the dataset is simulated, the problem at hand is to predict patient medical outcomes based on various health and personal metrics.

### 2. **Data Collection**
   The dataset was sourced from Kaggle, so no further data collection was necessary.

### 3. **Data Cleaning**
   Cleaning the data ensures that it is ready for analysis. This includes handling any missing values, correcting data types, and normalizing certain features for better model performance.

### 4. **Exploratory Data Analysis (EDA)**
   During this step, I analyzed the distributions and relationships between variables to identify potential patterns or insights. Visualizations such as pair plots were created to understand correlations between variables like BMI, blood pressure, and the target outcome (medical condition).

### 5. **Feature Engineering**
   I transformed categorical features into numerical representations using one-hot encoding and standardized the numeric features. Feature scaling was applied using `StandardScaler` to prepare the dataset for machine learning models, ensuring consistency across features.

### 6. **Model Selection and Training**
   Various machine learning models were trained on the dataset, including:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)

### 7. **Model Evaluation**
   Each model was evaluated using the following metrics:
   - **Accuracy:** Measures the proportion of correct predictions.
   - **Precision:** Measures the accuracy of positive predictions.
   - **Recall:** Measures the ability to identify positive cases.
   - **F1 Score:** The harmonic mean of precision and recall, balancing both metrics.
   
   A bar chart was generated to compare the performance of different models.

### 8. **Insights and Results**

The insights derived from the dataset help us understand the relationships between patient metrics and medical outcomes. For instance, certain variables like BMI, blood pressure, and age were found to be important predictors of patient conditions. Random Forest and Gradient Boosting models performed the best in terms of accuracy and recall, suggesting that ensemble models are particularly effective in healthcare data classification tasks.

### 9. **Why These Insights Are Important**

Healthcare is a domain where data-driven decisions can significantly impact patient outcomes. Being able to predict medical conditions based on a combination of health metrics and personal information allows healthcare providers to prioritize care, reduce risk, and improve resource allocation. For example, identifying at-risk patients for conditions like diabetes or cardiovascular diseases through early prediction models can help implement preventive care strategies, ultimately improving patient outcomes.

### 10. **Future Steps**
   - **Model Deployment:** While this project focused on training and evaluating models, future work could include deploying the best-performing model into a real-world healthcare environment, potentially integrating it into hospital systems for live predictions.
   - **Monitoring and Maintenance:** Ongoing evaluation of model performance is crucial for maintaining accuracy as new data becomes available.
   - **Reporting and Communication:** Clear communication of results to healthcare stakeholders can help in translating model insights into actionable decisions.

## Conclusion

This project demonstrates the end-to-end process of working with healthcare data, from cleaning and preprocessing to building and evaluating machine learning models. The insights gained from the analysis emphasize the importance of data-driven decisions in healthcare, where accurate predictions can lead to better patient care and more efficient resource management.

## How to Use

1. **Clone this repository**:
    ```bash
    git clone https://github.com/your-username/Healthcare-Dataset-Analysis.git
    cd Healthcare-Dataset-Analysis
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the notebook** in your local environment or use Google Colab:
    ```bash
    jupyter notebook HealthcareDataset.ipynb
    ```

4. **Explore the results**: The models will be trained and evaluated automatically. You can visualize the performance of each model and inspect the metrics to understand the predictions.

## Contact

For any questions or suggestions, feel free to contact me:
- **Name**: Kendall Dwyre
- **Email**: ksdwyre@gmail.com
- **LinkedIn**: [Kendall Dwyre LinkedIn](https://www.linkedin.com/in/kendall-dwyre/)
- **GitHub**: [Kendall Dwyre GitHub](https://github.com/kendall-dwyre/)

---

