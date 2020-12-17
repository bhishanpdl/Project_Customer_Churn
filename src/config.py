import os

# data
dat_dir = os.path.join('..','data')

path_data = os.path.join(dat_dir, 'raw', 'telco_customer_churn.csv')
path_data_raw = os.path.join(dat_dir, 'raw', 'telco_customer_churn.csv')
path_data_clean = os.path.join(dat_dir, 'processed', 'data_cleaned_encoded.csv')

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