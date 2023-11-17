# Evaluation_Metrics_for_Classification

![image](https://github.com/ChidimmaIdika/Evaluation_Metrics_for_Classification/assets/137975543/8630faaf-e442-4f51-bc37-cfd405a3ffcf)

---
*This project explores **Evaluation Metrics** to gauge the quality and performance of a previously built churn prediction model*

---

# Table of Contents
- [Introduction](#introduction)
- [Session Overview](#session-overview)
- [Accuracy and Dummy Model](#accuracy-and-dummy-model)
- [Confusion Table](#confusion-table)
- [Precision and Recall](#precision-and-recall)
- [ROC Curves](#roc-curves)
- [ROC AUC](#roc-auc)
- [Cross-Validation](#cross-validation)
- [Summary](#summary)

---
## Introduction
In a previous project [Churn Prediction Project](https://github.com/ChidimmaIdika/Churn-Prediction-Project-using-Machine-Learning.git), I built a predictive model for customer churn using a dataset from Telco. This project aims to assess the quality of a previously built model. In this post, I will delve into various evaluation metrics to gauge the performance of the churn prediction model.

## Session Overview
**Dataset:** Telco Customer Churn Dataset    
**Metric:** A function that compares predictions with actual values, providing a single numerical indicator of prediction quality.    
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data preprocessing
df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

# Train-test split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

# Reset indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Target variables
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

# Drop target variable from features
del df_train['churn']
del df_val['churn']
del df_test['churn']

# Define feature sets
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [...  # List of categorical features]

# Feature engineering using DictVectorizer
dv = DictVectorizer(sparse=False)
train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Validation set predictions
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)
y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
```

## Accuracy and Dummy Model
I will start by evaluating accuracy and comparing it with a dummy baseline.   
```python
from sklearn.metrics import accuracy_score

# Check accuracy at different thresholds
thresholds = np.linspace(0, 1, 21)
scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)
```

## Confusion Table
Understanding different types of errors and correct decisions using a confusion matrix.   
```python
# Confusion matrix calculations
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()
fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

confusion_matrix = np.array([[tn, fp], [fn, tp]])
confusion_matrix_percentage = (confusion_matrix / confusion_matrix.sum()).round(2)
```

## Precision and Recall
Precision and recall metrics provide insights into the model's performance.
```python
# Precision
p = tp / (tp + fp)
print("Precision:", p)

# Recall
r = tp / (tp + fn)
print("Recall:", r)
```

## ROC Curves
Receiver Operating Characteristic (ROC) curves visualize the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR).
```python
# TPR and FPR calculations
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

# Plot ROC curve
plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.xlabel('Threshold')
plt.legend();
```

## ROC AUC
Area Under the ROC Curve (AUC) is a useful metric for model evaluation.
```python
# Calculate ROC AUC
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)
```

## Cross-Validation
Evaluate the model using K-Fold Cross-Validation for robust performance estimation.
```python
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

n_splits = 5

# K-Fold Cross-Validation
for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))
```

## Summary
- **Metric:** A single number describing model performance.
- **Accuracy:** Fraction of correct answers, but can be misleading.
- **Precision and Recall:** Less misleading with class imbalance.
- **ROC Curve:** Evaluates performance across thresholds, suitable for imbalanced datasets.
- **ROC AUC:** Area under the ROC curve, a useful metric.
- **K-Fold CV:** Provides a more reliable estimate of performance.

---
*View notebook **[Here](https://github.com/ChidimmaIdika/Evaluation_Metrics_for_Classification/blob/Chidimma/Evaluation%20Metrics%20for%20Classification.ipynb)***     


*In further projects, these insights can be used to fine-tune models and improve its predictive capabilities.*    


*Stay tuned for more updates on churn prediction!*

---
