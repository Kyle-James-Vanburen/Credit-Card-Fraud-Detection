# Credit Card Fraud Detection 2023

## About the Dataset:
This dataset encompasses credit card transactions conducted by European cardholders throughout the year 2023. With a vast repository of over 550,000 records, the dataset has undergone anonymization procedures to safeguard the identities of the cardholders involved.

### Primary Objective:
The primary objective of this dataset is to facilitate the development of fraud detection algorithms and models to identify potentially fraudulent transactions.

### Transaction Type Analysis:
Analyze whether certain types of transactions are more prone to fraud than others.

### Key Features:
- **id:** Unique identifier for each transaction.
- **V1-V28:** Anonymized features representing various transaction attributes (e.g., time, location, etc.).
- **Amount:** The transaction amount.
- **Class:** Binary label indicating whether the transaction is fraudulent (1) or not (0).

## Dataset Source

Analysis is based on the "Credit Card Fraud Detection Dataset 2023" sourced from Kaggle. Access the dataset here: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

## Importing Dataset to Google Colab:
```python
import gdown

# Paste the file ID obtained from the provided Google Drive link below
file_id = '1ipjEJoZQMa5AMaaoJ1y3kv988jkC_uKL'

# Define the output file name
output_file = 'creditcard_2023.csv'

# Download the file from Google Drive
gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)
```

## Required Libraries:
```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
```

## Loading and Exploring the Dataset:
```python
# Load dataset to Pandas DataFrame
credit_card_data = pd.read_csv('creditcard_2023.csv')

# Display first five rows of the dataset
print(credit_card_data.head())

# Display dataset information
print(credit_card_data.info())

# Check for missing values
print(credit_card_data.isnull().sum())

# Distribution of non-fraudulent & fraudulent transactions
print(credit_card_data['Class'].value_counts())
```

## Data Analysis and Visualization:
Explore various aspects of the dataset such as statistical measures, transaction distribution, and feature relationships using visualizations like scatter plots, box plots, and pair plots.

## Data Preprocessing:
Separate the data into features (X) and labels (Y). Split the data into training and testing sets.

```python
# X contains features, Y contains labels
X = credit_card_data.drop(columns='Class', axis=1)
Y = credit_card_data['Class']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

## Building and Evaluating the Model:
```python
# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model with training data
model.fit(X_train, Y_train)

# Make predictions on training and testing data
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)

# Evaluate the model performance
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)

training_data_precision = precision_score(Y_train, X_train_prediction)
testing_data_precision = precision_score(Y_test, X_test_prediction)

training_data_recall = recall_score(Y_train, X_train_prediction)
testing_data_recall = recall_score(Y_test, X_test_prediction)

training_data_f1 = f1_score(Y_train, X_train_prediction)
testing_data_f1 = f1_score(Y_test, X_test_prediction)
```

## Model Performance:
- **Accuracy on Training Data:** {training_data_accuracy}
- **Accuracy on Testing Data:** {testing_data_accuracy}
- **Precision on Training Data:** {training_data_precision}
- **Precision on Testing Data:** {testing_data_precision}
- **Recall on Training Data:** {training_data_recall}
- **Recall on Testing Data:** {testing_data_recall}
- **F1 Score on Training Data:** {training_data_f1}
- **F1 Score on Testing Data:** {testing_data_f1}

## Visualization Insights:
- **Scatter Plot:** Class Separation: Fraud vs. Non-Fraud
- **Box Plot:** Fraud vs. Non-Fraud Transaction Distribution
- **Pair Plot:** Feature Relationships: Fraud vs. Non-Fraud

## Summary of Findings:
- The developed machine learning model demonstrates robust performance with approximately 79.17% accuracy on training data and 78.94% accuracy on testing data.
- Visualization insights highlight distinct patterns between non-fraudulent and fraudulent transactions.

## Call to Action:
- **Refine Detection Algorithms:** Develop cutting-edge models capable of identifying complex fraudulent patterns.
- **Implement Real-Time Surveillance:** Integrate systems enabling instantaneous transaction monitoring.
- **Continuous Monitoring and Adaptation:** Dynamically adjust detection techniques to match evolving fraud tactics.
- **Collaborate and Share Insights:** Foster collaborations with fellow data scientists for collective intelligence.
- **Empower Customer Vigilance:** Launch awareness campaigns, educating customers about secure online practices and prevalent fraud schemes.

---

### Contact

If you have any questions or need further assistance, please feel free to contact the project maintainer at vanburen.kyle@yahoo.com.

Feel free to modify and expand upon this README to provide additional details about your project!

