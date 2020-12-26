# Import libraries
import os
import random
import time
time_start_notebook = time.time()

# data science
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

# special
import pycaret
import pycaret.classification as pyc

# local imports
import config
import util

# machine learning
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn import metrics as skmetrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold


# warnings
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize.linesearch import LineSearchWarning

warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter('ignore', category=LineSearchWarning)

# random state
SEED = config.SEED
seed = SEED
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)

# configs
path_data_train = config.path_data_train
path_data_test = config.path_data_test
index_name = 'customerID'
target_name = 'Churn'


# ===================== Functions=====================
def profit(y_true, y_pred):
    tn, fp, fn, tp = skmetrics.confusion_matrix(y_true, y_pred).ravel()
    loss = 400 * tp - 200 * fn - 100 * fp
    return profit


# =================== load the data
df_train = pd.read_csv(path_data_train)
df_test = pd.read_csv(path_data_test)
print('df_train', df_train.shape)
print('df_test', df_test.shape)

ser_ids_train = df_train.pop(index_name)
ser_ids_test = df_test.pop(index_name)

m = {'Yes': 1, 'No': 0}

# ser_ytrain = df_train.pop(target_name).map(m)
# ser_ytest = df_test.pop(target_name).map(m)

ser_ytrain = df_train[target_name].map(m)
ser_ytest = df_test[target_name].map(m)

ytrain = np.array(ser_ytrain).flatten()
ytest = np.array(ser_ytest).flatten()

# ========================== data processing
df_train = util.clean_data(df_train)
df_test = util.clean_data(df_test)

# ========================== scaling numerical features
cols_scale = ['tenure', 'MonthlyCharges', 'TotalCharges',
              'Contract_TotalCharges_mean_diff',
              'PaymentMethod_MonthlyCharges_mean_diff']

# Scale the relevant columns
transformer = ColumnTransformer([('yeo_johnson',
                                  PowerTransformer(), cols_scale)],
                                remainder='passthrough')
transformer.fit(df_train)

Xtrain = transformer.transform(df_train)
Xtest = transformer.transform(df_test)


# ================================================================
exp = pyc.setup(df_train,target_name,
                train_size=0.8,
                session_id=SEED,
                preprocess = True,
                categorical_features = None,
                ordinal_features = None,
                high_cardinality_features = None,
                numeric_features = None,
                date_features = None,
                ignore_features = None,
                normalize = False,
                data_split_stratify = True,
                silent=True,
                profile=False,
                log_experiment=False,
                polynomial_features=True,
                # fix_imbalance=True, # gives attribute error

                )
# add metrics
pyc.add_metric('logloss', 'LogLoss', skmetrics.log_loss,
               greater_is_better=False)

model_name = 'lda'
model = pyc.create_model(model_name,verbose=False)
mean_row = pyc.pull().loc['Mean'].values
print(mean_row)


# ========================= time taken
time_taken = time.time() - time_start_notebook
h, m = divmod(time_taken, 60 * 60)
print('Time taken to run whole notebook: {:.0f} hr ' \
      '{:.0f} min {:.0f} secs'.format(h, *divmod(m, 60)))
