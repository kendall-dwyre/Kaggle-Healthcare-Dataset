# Kaggle-Healthcare-Dataset
This dataset is from Kaggle (https://www.kaggle.com/datasets/prasad22/healthcare-dataset).  It is simulated data.  The purpose is to practice, refine, and show my machine learning skills.  I really appreciate the heart behind the inspiration of the dataset, so I thought I would include it below:

    The inspiration behind this dataset is rooted in the need for practical and diverse healthcare data for educational and research purposes. Healthcare data is often sensitive and subject   to privacy regulations, making it challenging to access for learning and experimentation. To address this gap, I have leveraged Python's Faker library to generate a dataset that mirrors the structure and attributes commonly found in healthcare records. By providing this synthetic data, I hope to foster innovation, learning, and knowledge sharing in the healthcare analytics domain.

The purpose of this repository is to demonstrate machine learning skills by going through a dataset from online.

The dataset that I am working with is called "Pima Indians Diabetes Database" from Kaggle. You can find a link to it here: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download

Explanation about the dataset from Kaggle:

"This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage."

The objective of this repository is to demonstrate some machine learning skills; however, I will be going through a typical work flow that a data scientist would go through when working with new data. The steps are as follows:

1.) Define the Problem

2.) Collect Data

3.) Data Cleaning

4.) Explortory Analysis

5.) Feature Engineering

6.) Model Selection / Model Training

7.) Model Evaluation

8.) Model Deployment

9.) Monitor and Maintenance

10.) Communication and Reporting

Given that I gathered this data from Kaggle, steps 1.) and 2.) will be skipped. If we were starting scratch, we would begin by asking a question and gathering appropriate data.

Data Cleaning

We are looking for anything that may throw an issue - such as Null Values, NA's, or simply put bad data. In our case, it seems to be that this dataset was prepared before loading it onto Kaggle, but it's still good to go through the process and make sure nothing is overlooked.

Below is an example of some code that I wrote to ascertain if there are any null values (which there are none):
