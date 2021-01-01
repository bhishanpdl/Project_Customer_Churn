import os
from sklearn import metrics as skmetrics


# data
dat_dir = os.path.join('..','data')


path_data_raw = os.path.join(dat_dir, 'raw', 'telco_customer_churn.csv')
path_data_train = os.path.join(dat_dir, 'raw', 'train.csv')
path_data_test = os.path.join(dat_dir, 'raw', 'test.csv')
compression = None

report_dir = os.path.join('..','reports')
path_report_pandas_profiling = os.path.join(report_dir,'report_pandas_profiling.html')
path_report_sweetviz = os.path.join(report_dir,'report_sweetviz.html')

# params
model_type = 'classification'
target = 'Churn'
train_size = 0.8
test_size = 1-train_size
SEED = 100

cols_num = ['TotalCharges','tenure', 'MonthlyCharges']

cols_scale = ['tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract_TotalCharges_mean_diff',
            'PaymentMethod_MonthlyCharges_mean_diff']


#================= LogisticRegressionCV
def custom_loss(y_true, y_pred):
    tn, fp, fn, tp = skmetrics.confusion_matrix(y_true,y_pred).ravel()
    loss = 400*tp - 200*fn - 100*fp
    return loss

scoring = skmetrics.make_scorer(custom_loss, greater_is_better=True)
solver,penalty,l1_ratios = 'lbfgs','l2',None
Cs = [10**-i for i in range(5)]


params_lr = dict(
    random_state = SEED,
    scoring      = scoring, # f1,roc_auc, recall
    n_jobs       = -1,
    solver       = solver, # lbfgs, saga
    penalty      = penalty, # l1 l2 elasticnet(only saga)
    class_weight = 'balanced', # None
    max_iter     = 5000,
    Cs           = Cs,
    l1_ratios    = l1_ratios
    )

params_lgb1 = {
    'colsample_bytree' : 0.48709147773176875,
    'reg_alpha'        : 0,
    'reg_lambda'       : 0.04107179981116397,
    'learning_rate'    : 0.7991073935301635,
    'max_bin'          : 456,
    'max_depth'        : 13,
    'min_child_weight' : 0.03465070069968173,
    'min_data_in_bin'  : 83,
    'min_child_samples': 281,
    'min_split_gain'   : 1.49,
    'n_estimators'     : 5000,
    'scale_pos_weight' : 9,
    'subsample'        : 0.7857616414334825
    }

params_lgb = {'colsample_bytree': 0.5262112905471411, 'learning_rate': 0.5199550564232394, 'max_bin': 43, 'max_depth': 13, 'min_child_samples': 306, 'min_child_weight': 0.0019054209140305946, 'min_data_in_bin': 187, 'min_split_gain': 3.5100000000000002, 'n_estimators': 4600, 'num_leaves': 2, 'reg_alpha': 2.3316343178775068e-07, 'reg_lambda': 0, 'scale_pos_weight': 5, 'subsample': 0.7742077850067172}