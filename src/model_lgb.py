# Import libraries
import time
time_start_notebook = time.time()

import argparse
import pandas as pd

# local imports
import config
import util

# random state
import os
import random
import numpy as np

# machine learning
from sklearn import metrics as skmetrics
from sklearn.model_selection import StratifiedKFold

# boosting
import lightgbm as lgb


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

model_name = 'lgb'

# warnings
import warnings
warnings.simplefilter('ignore')
#===================================================

def get_profit(y_true, y_pred):
    tn, fp, fn, tp = skmetrics.confusion_matrix(y_true, y_pred).ravel()
    loss = 400 * tp - 200 * fn - 100 * fp
    return loss


def get_train_test_data():
    df_train = pd.read_csv(path_data_train)
    df_test = pd.read_csv(path_data_test)
    print('df_train', df_train.shape)
    print('df_test ', df_test.shape)

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
    Xtrain = df_train.to_numpy()

    return Xtrain,ytrain,df_test,ytest,ser_ids_train,ser_ytrain


def modelling(Xtrain, ytrain,ser_ids_train,ser_ytrain,df_test,ytest,params):
    df_preds = pd.DataFrame({index_name: [np.nan] * len(Xtrain),
                             'ytrain': -1,
                             'fold': -1,
                             f'pred_{model_name}': -1,
                             f'prob_{model_name}': np.nan
                             })

    pred_name = 'pred_' + model_name
    prob_name = 'prob_' + model_name
    skf = StratifiedKFold(shuffle=True, random_state=SEED, n_splits=5)
    profits = []
    for fold, (idx_tr, idx_vd) in enumerate(skf.split(Xtrain, ytrain)):
        # index and target
        df_preds.loc[idx_vd, index_name] = ser_ids_train.loc[idx_vd]
        df_preds.loc[idx_vd, 'ytrain'] = ser_ytrain.loc[idx_vd]
        df_preds.loc[idx_vd, 'fold'] = fold

        # data for out of fold prediction
        Xtr, ytr = Xtrain[idx_tr], ytrain[idx_tr]
        Xvd, yvd = Xtrain[idx_vd], ytrain[idx_vd]

        # out of fold prediction
        model = lgb.LGBMClassifier(**params)
        model.fit(Xtr, ytr)
        vdpred = model.predict(Xvd)
        vdprob2d = model.predict_proba(Xvd)
        vdprob = vdprob2d[:, 1]
        profit = get_profit(yvd, vdpred)
        print(f"validation fold {fold} profit = {profit:,d}")
        profits.append(profit)

        df_preds.loc[idx_vd,pred_name] = vdpred
        df_preds.loc[idx_vd, prob_name] = vdprob
    #=================================================
    #========= prediction on test set=================
    print('='*40)
    print(f"mean validation profit   = {np.mean(profits):,.0f} with std of ${np.std(profits):,.0f}")
    model = lgb.LGBMClassifier(**params)
    model.fit(Xtrain, ytrain)
    ypreds = model.predict(df_test)
    yprobs2d = model.predict_proba(df_test)
    profit = get_profit(ytest, ypreds)
    print(f"\ntest set profit = {profit:,d}")
    print('test set confusion matrix')
    print(skmetrics.confusion_matrix(ytest,ypreds))
    print(skmetrics.classification_report(ytest, ypreds))
    util.model_eval_bin(model_name, ytest, ypreds, yprobs2d, show_plots=False,disp=False)

    return df_preds,pred_name,prob_name,profits


def get_prediction(df_preds, pred_name, prob_name):
    df_preds['ytrain'] = df_preds['ytrain'].astype(np.int8)
    df_preds['fold'] = df_preds['fold'].astype(np.int8)
    df_preds[pred_name] = df_preds[pred_name].astype(np.int8)
    df_preds.to_csv(f'../predictions/tr_preds_{model_name}.csv', index=False)
    print()
    print(df_preds.head())


def time_taken():
    time_taken = time.time() - time_start_notebook
    h, m = divmod(time_taken, 60 * 60)
    print('Time taken to run whole notebook: {:.0f} hr ' \
      '{:.0f} min {:.0f} secs'.format(h, *divmod(m, 60)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params', help='parameters name',
                        type=str, required=False, default='params_lgb')

    args = parser.parse_args()
    params_name = args.params

    params = getattr(config, params_name)
    params['random_state'] = config.SEED
    params['silent'] = True


    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Xtrain, ytrain, df_test, ytest, ser_ids_train, ser_ytrain = get_train_test_data()
        df_preds, pred_name, prob_name, scores = modelling(
            Xtrain, ytrain,ser_ids_train,ser_ytrain,df_test,ytest,params)
        get_prediction(df_preds, pred_name, prob_name)
        time_taken()







