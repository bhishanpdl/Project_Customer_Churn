#!/usr/bin/env python

__doc__ = """
Author: Bhishan Poudel

We must start a project with train test split.
We do all the modelling on train data and then test the
model performance on test data in the end.

When training the model, we should always be careful about data leakage.
To avoid the data leakage we can check following:
- look for modified target feature, e.g log(target)

"""

# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# local imports
import util
import config

# random state
SEED = 100 # keep it fixed here, do not use from config file
np.random.seed(SEED)

# params
path_data_raw   = config.path_data_raw
path_data_train = config.path_data_train
path_data_test  = config.path_data_test
train_size      = config.train_size
compression     = config.compression

train_size      = config.train_size
model_type      = config.model_type
target          = config.target

# Load the data
df = pd.read_csv(path_data_raw)

# regression vs classification
if model_type == 'regression':
    stratify = None
if model_type == 'classification':
    stratify=df[target]

# train test split
df_train, df_test = train_test_split(
    df,
    train_size=train_size,
    random_state=SEED,
    stratify=stratify
    )

# prints
print(f"df       : {df.shape}")
print(f"df_train : {df_train.shape}")
print(f"df_test  : {df_test.shape}")

if model_type == 'classification':
    print('target distribution in df')
    print(df[target].value_counts(normalize=True))

    print('target distribution in train')
    print(df_train[target].value_counts(normalize=True))

    print('target distribution in test')
    print(df_test[target].value_counts(normalize=True))

# write files
df_train.to_csv(path_data_train,index=False)
df_test.to_csv(path_data_test,index=False)