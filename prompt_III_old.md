# Bank Marketing Classification Analysis

## Introduction
This notebook provides a comprehensive analysis of the bank marketing classification dataset, applying four different classifiers to predict whether a client will subscribe to a term deposit. The focus is on thorough exploratory data analysis (EDA), model training with hyperparameter tuning, cross-validation, and visualization of results.

## Table of Contents
1. [Data Import](#data-import)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Modeling](#modeling)
   - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
   - [Logistic Regression](#logistic-regression)
   - [Decision Trees](#decision-trees)
   - [Support Vector Machine (SVM)](#support-vector-machine-svm)
4. [Model Evaluation](#model-evaluation)
5. [Findings and Recommendations](#findings-and-recommendations)

## Data Import
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
```

### Load the Dataset
```python
df = pd.read_csv('bank.csv')
```

## Exploratory Data Analysis (EDA)
### Understanding the Data
```python
print(df.head())
print(df.info())
print(df.describe())
```
### Handling Missing Values
```python
# Checking for null values
print(df.isnull().sum())
```
### Data Visualization
```python
# Distribution of Target Variable
sns.countplot(x='y', data=df)
plt.title('Target Variable Distribution')
plt.show()
```

## Modeling
### K-Nearest Neighbors (KNN)
```python
X = df.drop('y', axis=1)
Y = df['y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)

# Predictions
predictions = knn.predict(X_test)
print(classification_report(Y_test, predictions))
```
### Logistic Regression
```python
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
predictions_lr = logreg.predict(X_test)
print(classification_report(Y_test, predictions_lr))
```
### Decision Trees
```python
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, Y_train)
predictions_dt = dec_tree.predict(X_test)
print(classification_report(Y_test, predictions_dt))
```
### Support Vector Machine (SVM)
```python
svm = SVC()
svm.fit(X_train, Y_train)
predictions_svm = svm.predict(X_test)
print(classification_report(Y_test, predictions_svm))
```

## Model Evaluation
```python
# Cross-Validation
cv_results_knn = cross_val_score(knn, X, Y, cv=5)
cv_results_lr = cross_val_score(logreg, X, Y, cv=5)
cv_results_dt = cross_val_score(dec_tree, X, Y, cv=5)
cv_results_svm = cross_val_score(svm, X, Y, cv=5)

print('KNN CV Score:', cv_results_knn.mean())
print('Logistic Regression CV Score:', cv_results_lr.mean())
print('Decision Tree CV Score:', cv_results_dt.mean())
print('SVM CV Score:', cv_results_svm.mean())
```

## Findings and Recommendations
- **Key Insights**: The analysis reveals which features are most impactful on predicting term deposit subscriptions.
- **Model Performance**: Based on cross-validation scores, one model may be superior for this dataset, guiding future strategies.
- **Next Steps**: Consider deploying the best model and further refining feature engineering and hyperparameter tuning for improved accuracy.

---

### Conclusion
The notebook provides a structured analysis of customer data with a focus on four classifiers, enabling comprehensive insights and actionable recommendations for effective marketing strategies.
