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

#================================ LightGBM ==================================================================
# this is default parameters
params_lgb0 = dict(
    boosting_type     = 'gbdt',
    num_leaves        = 31,
    max_depth         = -1,
    learning_rate     = 0.1,
    n_estimators      = 100,
    subsample_for_bin = 200000,
    objective         = None,
    class_weight      = None,
    min_split_gain    = 0.0,
    min_child_weight  = 0.001,
    min_child_samples = 20,
    subsample         = 1.0,
    subsample_freq    = 0,
    colsample_bytree  = 1.0,
    reg_alpha         = 0.0,
    reg_lambda        = 0.0,
    random_state      = SEED,
    n_jobs            = -1,
    silent            = True,
    importance_type   = 'split'
)

params_lgb1 = {'colsample_bytree': 0.707665744307114, 'learning_rate': 0.7858437768791948, 'max_bin': 248,
               'max_depth': 6, 'min_child_samples': 381, 'min_child_weight': 1.65375790265872e-05,
               'min_data_in_bin': 139, 'min_split_gain': 2.3000000000000003, 'n_estimators': 2200,
               'num_leaves': 525, 'reg_alpha': 0, 'reg_lambda': 0,
               'scale_pos_weight': 7,'subsample': 0.7273890631561498}

params_lgb2 = {'colsample_bytree': 0.8757770860663306, 'learning_rate': 0.8718058384285751, 'max_bin': 71,
               'max_depth': 10, 'min_child_samples': 337, 'min_child_weight': 0.0038481016662428053,
               'min_data_in_bin': 196, 'min_split_gain': 3.75, 'n_estimators': 4400, 'num_leaves': 18,
               'reg_alpha': 3.552634042044225e-07, 'reg_lambda': 5.754838317539429e-05,
               'scale_pos_weight': 7,'subsample': 0.6747518226204151}

params_lgb3 = {'colsample_bytree': 0.5262112905471411, 'learning_rate': 0.5199550564232394, 'max_bin': 43,
               'max_depth': 13, 'min_child_samples': 306, 'min_child_weight': 0.0019054209140305946,
               'min_data_in_bin': 187, 'min_split_gain': 3.5100000000000002, 'n_estimators': 4600,'num_leaves': 2,
               'reg_alpha': 2.3316343178775068e-07, 'reg_lambda': 0,
               'scale_pos_weight': 5, 'subsample': 0.7742077850067172}

params_lgb4 = {'colsample_bytree': 0.7614216209026772, 'learning_rate': 0.816821855221229, 'max_bin': 114,
               'max_depth': 27, 'min_child_samples': 411, 'min_child_weight': 2.1524026408064625e-05,
               'min_data_in_bin': 71, 'min_split_gain': 3.4, 'n_estimators': 350, 'num_leaves': 466,
               'reg_alpha': 7.08190801243234e-05, 'reg_lambda': 0,
               'scale_pos_weight': 7,'subsample': 0.571824428670002}

params_lgb = {'colsample_bytree': 0.7614216209026772, 'learning_rate': 0.816821855221229, 'max_bin': 114,
               'max_depth': 27, 'min_child_samples': 411, 'min_child_weight': 2.1524026408064625e-05,
               'min_data_in_bin': 71, 'min_split_gain': 3.4, 'n_estimators': 350, 'num_leaves': 466,
               'reg_alpha': 7.08190801243234e-05, 'reg_lambda': 0,
               'scale_pos_weight': 7,'subsample': 0.571824428670002}