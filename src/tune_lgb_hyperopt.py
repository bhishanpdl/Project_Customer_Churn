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
import hyperopt
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample
from pathlib import Path
import json



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


path_trials = f'../artifacts/hyperopt_trials_{model_name}.joblib'

#================= parameters
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
space_lgb_hyperopt = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),

    'max_depth': scope.int(hp.quniform('max_depth', 2, 32, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 100, 5000, 50)),
    'max_bin': scope.int(hp.quniform('max_bin', 8, 512, 1)),

    'min_child_samples': scope.int(hp.quniform('min_child_samples', 1, 512, 1)),  # min_data_in_leaf
    'min_data_in_bin': scope.int(hp.quniform('min_data_in_bin', 1, 512, 1)),  # default = 3
    'num_leaves': scope.int(hp.quniform('num_leaves', 2, 1000, 1)),
    # default num_leaves = 31 and 2**max_depth > num_leaves
    # constraint 1 < num_leaves <= 131072

    'scale_pos_weight': hp.randint('scale_pos_weight', 1, 50),
    # use only one of scale_pos_weight or class_weight
    # 'class_weight': hp.choice('class_weight', [None, 'balanced']),

    # 'subsample_freq'   : hp.randint('subsample_freq', 0, 50),
    # WARNING: This gave me worse result, dont use this.
    # perform bagging at every k iteration. Every k-th iteration,
    # LightGBM will randomly select bagging_fraction * 100 %
    # of the data to use for the next k iterations
    # to enable bagging, bagging_fraction should be set to value
    # smaller than 1.0 as well

    'subsample': hp.uniform('subsample', 0.5, 1),  # bagging_fraction
    'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 1.0),  # feature_fraction

    # regularization
    'reg_alpha': hp.choice('reg_alpha', [0, hp.loguniform('reg_alpha_positive', -16, 2)]),  # lambda_l1
    'reg_lambda': hp.choice('reg_lambda', [0, hp.loguniform('reg_lambda_positive', -16, 2)]),  # lambda_l2

    'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
    'min_split_gain': hp.quniform('min_split_gain', 0.1, 5, 0.01),

}

INT_PARAMS = ['n_estimators','num_boost_round','num_leaves',
              'max_depth','max_bin',
              'min_child_samples','min_data_in_bin','subsample_freq']
#=======================================================================


def get_profit(y_true, y_pred):
    tn, fp, fn, tp = skmetrics.confusion_matrix(y_true, y_pred).ravel()
    loss = 400 * tp - 200 * fn - 100 * fp
    return loss

def time_taken():
    time_taken = time.time() - time_start_notebook
    h, m = divmod(time_taken, 60 * 60)
    print('Time taken to run whole notebook: {:.0f} hr ' \
      '{:.0f} min {:.0f} secs'.format(h, *divmod(m, 60)))

def lgb_objective_hyperopt_skf(params):
    global INT_PARAMS,  SEED, df_Xtrain_full, ser_ytrain_full

    for int_param in INT_PARAMS:
        # make integer if exist
        if int_param in params:
            params[int_param] = int(params[int_param])

    # num_leaves must be smaller than 2**max_depth
    if 'num_leaves' in params.keys() and params['max_depth'] > 0:
        params['num_leaves'] = int(min(params['num_leaves'], 2 ** params['max_depth'] - 1))

    # skf is more time-consuming but more stable.
    skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
    profits = []
    aucs = []
    num_best_rounds = []
    for idx_tr, idx_vd in skf.split(df_Xtrain_full, ser_ytrain_full):
        Xtr, Xvd = df_Xtrain_full.iloc[idx_tr], df_Xtrain_full.iloc[idx_vd]
        ytr, yvd = ser_ytrain_full[idx_tr], ser_ytrain_full.iloc[idx_vd]

        model = lgb.LGBMClassifier(random_state=SEED, **params)

        model.fit(Xtr, ytr,
                  eval_set=[(Xvd, yvd)],
                  verbose=0,
                  early_stopping_rounds=100)

        # get best round
        e = model.evals_result_
        k0 = list(e.keys())[0]
        k1 = list(e[k0].keys())[0]
        num_best_round = len(e[k0][k1])
        num_best_rounds.append(num_best_round)

        # model predictions
        vdpreds = model.predict(Xvd)
        yvd = yvd.to_numpy().ravel()

        auc_now = skmetrics.roc_auc_score(yvd, vdpreds)
        aucs.append(auc_now)

        profit_now = get_profit(yvd, vdpreds)
        profits.append(profit_now)

    # =============================================================
    profit = np.mean(profits)
    profit_std = np.std(profits)
    auc = np.mean(aucs)
    auc_std = np.std(aucs)
    num_best_round = np.max(num_best_rounds)

    # loss must be minimized, so we may need to use -ve sign.
    return {'loss': -profit, 'status': hyperopt.STATUS_OK,
            'profit': profit, 'profit_std': profit_std,
            'auc': auc, 'auc_std': auc_std,
            'num_best_round': num_best_round
            }

def get_train_data():
    df_train = pd.read_csv(path_data_train)
    ser_ids_train = df_train.pop(index_name)
    m = {'Yes': 1, 'No': 0}
    ser_ytrain_full = df_train.pop(target_name).map(m)
    df_Xtrain_full = util.clean_data(df_train)

    return df_Xtrain_full,ser_ytrain_full


if __name__ == '__main__':
    # parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_trials', help='number of trials to run',
                        type=int, required=False, default=1)
    args = parser.parse_args()
    n_trials = args.n_trials


    # get the data
    df_Xtrain_full, ser_ytrain_full = get_train_data()

    params_best = hyperopt.fmin(fn=lgb_objective_hyperopt_skf,
                                space=space_lgb_hyperopt,
                                algo=tpe.suggest,
                                max_evals=n_trials,
                                timeout=None,
                                trials=hyperopt.Trials(),
                                verbose=10,
                                show_progressbar=True,
                                rstate=np.random.RandomState(SEED)
                                )

    params_best = hyperopt.space_eval(space_lgb_hyperopt, params_best)
    print(params_best)

    path_params_best = f'../outputs/lgb_hyperopt_ntrials{n_trials}.json'
    if not Path('../outputs').exists():
        os.makedirs('../outputs')
    with open(path_params_best,'w') as fo:
        json.dump(params_best,fo)

    time_taken()


