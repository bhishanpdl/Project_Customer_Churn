# Import libraries
import time

time_start_notebook = time.time()
import pandas as pd

# local imports
import config
import util

# random state
import os
import random
import numpy as np

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

# settings
SEED = config.SEED
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
def custom_loss(y_true, y_pred):
    tn, fp, fn, tp = skmetrics.confusion_matrix(y_true, y_pred).ravel()
    loss = 400 * tp - 200 * fn - 100 * fp
    return loss


# =================== load the data
df_train = pd.read_csv(path_data_train)
df_test = pd.read_csv(path_data_test)
print('df_train', df_train.shape)
print('df_test', df_test.shape)

ser_ids_train = df_train.pop(index_name)
ser_ids_test = df_test.pop(index_name)

m = {'Yes': 1, 'No': 0}
ser_ytrain = df_train.pop(target_name).map(m)
ser_ytest = df_test.pop(target_name).map(m)
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

# ============================================ Modelling ==================
model_name = 'lrcv'

# Modelling: LogisticRegressionCV
params_lr = config.params_lr
# params_lr.update(dict(max_iter=5000))
# model = LogisticRegressionCV(**params_lr)
# model.fit(Xtrain, ytrain)
# ypreds   = model.predict(Xtest)
# yprobs2d = model.predict_proba(Xtest)
# profit = custom_loss(ytest,ypreds)
# print(f"profit = {profit:,d}")
# print(skmetrics.confusion_matrix(ytest, ypreds))

df_preds = pd.DataFrame({index_name: [np.nan]*len(Xtrain),
                         'ytrain': np.nan,
                         'fold': np.nan,
                         f'pred_{model_name}': np.nan
                         })


pred_name = 'pred_' + model_name
prob_name = 'prob_' + model_name
skf = StratifiedKFold(shuffle=True, random_state=SEED, n_splits=5)
scores = []
for fold, (idx_tr, idx_vd) in enumerate(skf.split(Xtrain, ytrain)):
    # index and target
    df_preds.loc[idx_vd, index_name] = ser_ids_train.loc[idx_vd]
    df_preds.loc[idx_vd, 'ytrain'] = ser_ytrain.loc[idx_vd]
    df_preds.loc[idx_vd, 'fold'] = fold

    # data for out of fold prediction
    Xtr, ytr = Xtrain[idx_tr], ytrain[idx_tr]
    Xvd, yvd = Xtrain[idx_vd], ytrain[idx_vd]

    # out of fold prediction
    model = LogisticRegressionCV(**params_lr)
    model.fit(Xtr,ytr)
    ypred = model.predict(Xvd)
    yprob2d = model.predict_proba(Xvd)
    yprob = yprob2d[:,1]
    profit = custom_loss(yvd,ypred)
    print(f"profit = {profit:,d}")
    scores.append(profit)

    df_preds.loc[idx_vd,pred_name] = ypred
    df_preds.loc[idx_vd, prob_name] = yprob

df_preds['ytrain'] = df_preds['ytrain'].astype(np.int8)
df_preds['fold'] = df_preds['fold'].astype(np.int8)
df_preds[pred_name] = df_preds[pred_name].astype(np.int8)
df_preds.to_csv(f'../predictions/tr_preds_{model_name}.csv',index=False)
print(df_preds.head())
print(f'mean profit for 5 fold training : {np.mean(scores):,.0f} with std of {np.std(scores):,.0f}')

# ========================= time taken
time_taken = time.time() - time_start_notebook
h, m = divmod(time_taken, 60 * 60)
print('Time taken to run whole notebook: {:.0f} hr ' \
      '{:.0f} min {:.0f} secs'.format(h, *divmod(m, 60)))
