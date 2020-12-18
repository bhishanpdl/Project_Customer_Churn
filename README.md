# Project Description
In this project I used the [Kaggle Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
data to determine whether the customer will churn (leave the company) or not.
I splitted the kaggle training data into train and test (80%/20%) and fitted
the models using train data and evaluated model results in test data.

# Data description
![](images/data_describe.png)
![](images/Churn.png)
![](images/Gender_Churn.png)
![](images/MonthlyCharges_Churn.png)
![](images/SeniorCitizen_vs_Churn.png)

# Data Processing
- Missing Value imputation for `TotalCharges` with 0.
- Label Encoding for features having 5 or less unique values.
- Binning Numerical Features.
- Combination of features. e.g `SeniorCitizen + Dependents`.
- Boolean Features. e.g. Does someone have Contract or not.
- Aggregation features. eg. Mean of `TotalCharges` per `Contract`.


# Sklearn Methods: Logistic Regression
- Used raw data with new features from EDA.
- Used SMOTE oversampling since data is imbalanced.
- Used `yeo-johnson` transformers instead of standard scaling since the numerical features were not normal.
- Tuned the model using [hyperband](https://github.com/thuijskens/scikit-hyperband) library.


```
   Accuracy	Precision	Recall	F1-score	AUC
LR	0.4450	0.3075	0.8717	0.4547	0.5812

[[301 734]
 [ 48 326]]
```

# Boosting: Xgboost
- Used minimal feature engineering.
- Get dummy featrues using `drop_first=False`
- Used xgb classifier with validation set and eval metric `aucpr`.

```


         Accuracy Precision  Recall  F1-score AUC
xgboost	0.7417   0.5083     0.8235  0.6286   0.7678

[[737 298]
 [ 66 308]]
```

# Modelling Pycaret
- Used detailed cleaned data.
- Pycaret uses gpu for xgboost and lightgbm in colab.
- Pycaret does not have model interpretation (SHAP) for non-tree based models.
- Simple model comparison gave naive bayes as the best model.
- Used additional metrics `MCC and LogLoss`.
- Used `tune-sklearn` algorithm to tune logistic regression.
- The model calibration in pycaret DID NOT improve the metric.

![](images/pycaret_compare_models.png)
![](images/pycaret_lr.png)


```
Pycaret Logistic Regression
=============================
            Accuracy Precision Recall    F1-score    AUC
pycaret_lr    0.7509 0.5199    0.8021    0.6309      0.7673

[[758 277]
 [ 74 300]]

Pycaret Naive Bayes
=============================
            Accuracy  Precision Recall    F1-score    AUC
pycaret_nb  0.7296    0.4943    0.8102    0.6140    0.7553
[[725 310]
 [ 71 303]]

Pycaret Xgboost (Takes long time, more than 1 hr)
==================================

                  Accuracy Precision Recall  F1-score  AUC
pycaret_xgboost    0.7601  0.5342    0.7513  0.6244    0.7573

[[790 245]
 [ 93 281]]
```

# Deep Learning models
- Used minimal data processing.
- Dropped `customerID` and `gender`.
- Imputed `TotalCharges` with 0.
- Created dummy variables from categorical features.
- Used standard scaling to scale the data.
- Used `class_weight` parameter to deal with imbalanced data.
- Tuned keras model with scikitlearn `GridSearchCV`


```python
Model parameters
{'activation': 'sigmoid',
 'batch_size': 128,
 'epochs': 30,
 'n_feats': 43,
 'units': (45, 30, 15)}

NOTE: The result changes each time even if I set SEED for everything.

      Accuracy  Precision Recall    F1-score    AUC
keras 0.6849    0.4422    0.7166    0.5469    0.6950
[[697 338]
 [106 268]]

```

# Model Comparion

```
This is a imbalanced binary classification.
The useful metrics are AUC and Recall.

- The pure xgboost model gave me the best area under the curve (AUCROC).
- The pure logistic regression model gave me the best Recall.

                 Accuracy   Precision Recall       F1-score       AUC
pycaret_lr       0.750887   0.519931  0.802139     0.630915  0.767253
xgboost_1        0.766501   0.544554  0.735294     0.625711  0.756536
xgboost          0.741661   0.508251  0.823529     0.628571  0.767803**
keras_2          0.734564   0.000000  0.000000     0.000000  0.500000
keras            0.684883   0.442244  0.716578     0.546939  0.695004
pycaret_xgboost  0.760114   0.534221  0.751337     0.624444  0.757311
pycaret_nb       0.729595   0.494290  0.810160     0.613982  0.755322
LR               0.444996   0.307547  0.871658**  0.454672  0.581240
```
