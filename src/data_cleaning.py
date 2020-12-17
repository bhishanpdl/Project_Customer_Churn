#!/usr/bin/env python

__doc__ = """
Author: Bhishan Poudel

Task: Clean the data of King County House price and create new features

- input: ../data/raw/kc_house_data.csv
- output: ../data/processed/data_cleaned_encoded.csv

- Created date features age and age_after_renovation.
- Created dummies for all categorical features
- Created log tranform features for features with large values
- Created new features like zipcode_houses (number of houses in that zipcode)
"""

# Imports
import numpy as np
import pandas as pd

# random state
SEED=100
np.random.seed(SEED) # we need this in each cell


# Load the data
df = pd.read_csv('../data/raw/kc_house_data.csv')


# Date time features
df['date'] = pd.to_datetime(df['date'])
df['yr_sales'] = df['date'].dt.year
df['age'] = df['yr_sales'] - df['yr_built']
df['yr_renovated2'] = np.where(df['yr_renovated'].eq(0), df['yr_built'], df['yr_renovated'])
df['age_after_renovation'] = df['yr_sales'] - df['yr_renovated2']


# Categorical Features
cols_str = ['waterfront', 'view', 'condition', 'grade','zipcode']
for c in cols_str:
    df[c] = df[c].astype(str)

cols_obj = df.select_dtypes(['object','category']).columns
cols_obj_small = ['waterfront', 'view', 'condition', 'grade']
# zipcode is related to house price, we may not want to drop it.
# there are 70 unique zipcode values, it will create too many dummies.
# one choice is taking top 5 or top 10 zipcodes
# we can choose top 10 zipcodes with largest house price.
# (or may be largest number of houses in that zipcode.)
most_expensive9_zipcodes = (df[['zipcode','price']]
                           .sort_values(['price','zipcode'])
                           .drop_duplicates('zipcode',keep='last')
                           .tail(9)
                           .zipcode
                           .values
                          )

# keep same zipcode for top 9 expensive and make all others as others
df['zipcode_top10'] = df['zipcode']
df.loc[~df['zipcode_top10'].isin(most_expensive9_zipcodes), 'zipcode_top10'] = 'others'

# we can also create new feature number of houses in that zipcode
df['zipcode_houses'] = df.groupby(['zipcode'])['price'].transform('count')


# Boolean data types
df['basement_bool'] = df['sqft_basement'].apply(lambda x: 1 if x>0 else 0)
df['renovation_bool'] = df['yr_renovated'].apply(lambda x: 1 if x>0 else 0)


# Numerical features binning
cols_bin = ['age','age_after_renovation']
df['age_cat'] = pd.cut(df['age'], 10, labels=range(10)).astype(str)
df['age_after_renovation_cat'] = pd.cut(df['age_after_renovation'], 10, labels=range(10))

# Create dummy variables from object and categories
cols_obj_cat = df.select_dtypes(include=[np.object, 'category']).columns
cols_dummy = ['waterfront', 'view', 'condition', 'grade',
              'zipcode_top10','age_cat', 'age_after_renovation_cat']

df_dummy = pd.get_dummies(df[cols_dummy],drop_first=False)
df_encoded = pd.concat([df,df_dummy], axis=1)

# Log transformation of large numerical values
cols_log = ['sqft_living', 'sqft_lot', 'sqft_above',
            'sqft_basement', 'sqft_living15', 'sqft_lot15']

for col in cols_log:
    df_encoded['log1p_' + col] = np.log1p(df[col])

# Drop unwanted columns
df.drop('id',inplace=True,axis=1)

# Save clean data
df_encoded.to_csv('../data/processed/data_cleaned_encoded.csv',
                  index=False,header=True)
