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