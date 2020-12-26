import os
import keras
from keras.constraints import maxnorm

# data
dat_dir = os.path.join('..','data')
data_path_raw = os.path.join(dat_dir, 'raw/customer_churn.csv')
data_path_clean = os.path.join(dat_dir, 'processed/data_cleaned_encoded.csv')

data_path_train = os.path.join(dat_dir, 'raw/train.csv')
data_path_test = os.path.join(dat_dir, 'raw/test.csv')
compression = None

# params
model_type = 'classification'
target = 'Churn'
train_size = 0.8
test_size = 1-train_size
SEED = 100


# params
#================================================
PARAMS_MODEL = {
    # first layer
    'L1_units'            : 16,
    'L1_act'              : 'relu',
    'L1_kernel_init'      : 'normal',
    'L1_kernel_reg'       : None,
    'L1_bias_reg'         : None,
    'L1_kernel_constraint': maxnorm,
    'L1_dropout'          : 0.2,

    # layer 2
    'L2_units'      : 8,
    'L2_act'        : 'relu',
    'L2_kernel_init': 'normal',
    'L1_kernel_constraint': maxnorm,
    'L2_kernel_reg' : None,
    'L2_bias_reg'   : None,
    'L2_dropout'    : 0.2,

    # NOTE: last layer is defined in model definition.

    # optimizer
    'optimizer': keras.optimizers.Adam(),
}

#=========================================================
METRICS = ['roc_auc' ] # use this name so that we can do sklearn gridsearch

#==========================================================
PARAMS_FIT = {'epochs': 100,
            'batch_size': 64,
            'patience': 50,
            'shuffle': True,
            'validation_split': None
            }
