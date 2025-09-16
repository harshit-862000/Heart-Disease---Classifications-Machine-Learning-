Heart Disease Prediction Using Machine Learning
Introduction
This project aims to predict the presence of heart disease in patients using various clinical features. It explores the heart disease dataset, conducts data preprocessing, and applies multiple machine learning classification algorithms to build robust predictive models. The project demonstrates exploratory data analysis (EDA), model comparison, and evaluation of healthcare data.

Features
Exploratory Data Analysis (EDA) with visualizations for feature and target distributions

Data cleaning, normalization, and categorical variable encoding

Implementation of multiple classifiers:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Naive Bayes

Decision Tree

Random Forest

Model performance evaluation using metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

Confusion matrix visualization and model comparison

Tech Stack
Python 3

Pandas, NumPy (Data manipulation)

Matplotlib, Seaborn (Visualization)

scikit-learn (Machine Learning models)

Jupyter Notebook / Kaggle Notebook

Workflow
Data Loading
Load and inspect the dataset (heart.csv).

EDA & Visualization
Analyze and visualize feature distributions and relationships.

Data Preprocessing

Handle categorical variables with one-hot encoding

Normalize numerical features

Model Training & Testing

Split data into training and testing sets

Train multiple classifiers and assess accuracy

Model Comparison & Evaluation

Compare test accuracies and confusion matrices

Identify the best-performing models

Results
Highest accuracy achieved was 88.52% using KNN and Random Forest classifiers.

Visualizations provide insights into feature impact on heart disease prediction.

Demonstrates practical steps for healthcare data analysis and ML implementation.

Usage
Clone the repo and run the notebook in Jupyter or Kaggle:

bash
git clone <your-repository-url>
Open and execute each cell in the notebook to reproduce results.

Dataset
heart.csv from Kaggle: Dataset Link

Contains features including age, sex, chest pain type, blood pressure, cholesterol, electrocardiographic results, heart rate, and more.

Contributing
Feedback and pull requests for improvements are welcome, especially regarding EDA and additional algorithms!
