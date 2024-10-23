# Credit-card-detection-using-Machin-learning

Here's an updated version of the **README**, including the information about the Kaggle dataset:

---

# Credit Card Fraud Detection using Machine Learning

## Overview

This Jupyter Notebook provides a comprehensive implementation of machine learning techniques for detecting fraudulent transactions in a credit card dataset. The goal is to build predictive models to distinguish between fraudulent and non-fraudulent transactions, utilizing various machine learning classifiers.

## Dataset

The dataset used in this project is the **Credit Card Fraud Detection** dataset from Kaggle, available [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data). The dataset contains 284,807 transactions, out of which 492 are fraudulent. It is highly imbalanced, with frauds accounting for only 0.172% of all transactions.

- **Features**: The dataset consists of 30 features, including time, amount, and anonymized features resulting from a PCA transformation.
- **Target**: The target variable is a binary classification label (`0` for non-fraudulent transactions and `1` for fraudulent transactions).

## Libraries Used

The notebook utilizes the following Python libraries:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations and working with arrays.
- **matplotlib & seaborn**: For data visualization.
- **plotly**: For interactive plots and visualizations.
- **scikit-learn**: For model training, evaluation, and metrics.
- **CatBoost, XGBoost**: Gradient boosting libraries for training efficient and scalable models.

## Main Sections

1. **Data Loading and Preprocessing**: 
   - Importing the credit card transaction dataset from Kaggle.
   - Handling missing data (if any) and normalizing or scaling features.
   - Splitting the dataset into training and testing sets using `train_test_split`.

2. **Exploratory Data Analysis (EDA)**:
   - Using visualizations (Seaborn, Matplotlib, and Plotly) to understand the distribution of features, especially focusing on fraudulent vs. non-fraudulent transactions.
   - Key plots include correlation heatmaps, bar charts, distribution plots, and box plots for identifying trends and patterns in the data.

3. **Modeling and Training**:
   - Implementing several machine learning models, including:
     - Random Forest Classifier
     - AdaBoost Classifier
     - CatBoost Classifier
     - XGBoost Classifier
   - Hyperparameter tuning and model training using the training dataset.
   - Evaluation of model performance with various metrics such as Accuracy, Precision, Recall, and ROC-AUC.

4. **Model Evaluation**:
   - Applying metrics like `roc_auc_score`, `precision_score`, `recall_score`, and `accuracy_score` to evaluate the models.
   - Comparing the performance of each classifier to determine the best-performing model for fraud detection.

## Visualizations

- The notebook includes interactive plots created using `plotly` to better visualize and analyze transaction patterns. These plots help in understanding how different features influence the classification of fraudulent vs. non-fraudulent transactions.

## How to Run

1. Install the required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn plotly scikit-learn catboost xgboost
   ```
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data).
3. Load the dataset into your Jupyter environment and follow the steps to preprocess the data, train models, and visualize the results.

## Conclusion

This project demonstrates the application of machine learning models for detecting credit card fraud. By comparing various classifiers, we aim to identify the best approach for detecting fraudulent transactions with high accuracy and reliability.

