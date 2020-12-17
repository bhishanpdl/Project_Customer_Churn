import os
import keras

# data
dat_dir = os.path.join('..','data')
data_path = os.path.join(dat_dir, 'raw/kc_house_data.csv')
data_path_raw = os.path.join(dat_dir, 'raw/kc_house_data.csv')
data_path_clean = os.path.join(dat_dir, 'processed/data_cleaned_encoded.csv')

data_path_train = os.path.join(dat_dir, 'raw/train.csv')
data_path_test = os.path.join(dat_dir, 'raw/test.csv')
compression = None

# params
model_type = 'regression'
target = 'price'
train_size = 0.8
test_size = 1-train_size
SEED = 100


# params
#===============================================================================
PARAMS_MODEL = {
    # first layer
    'L1_units'      : 80,
    'L1_act'        : 'tanh',
    'L1_kernel_init': 'normal',
    'L1_kernel_reg' : None,
    'L1_bias_reg'   : None,
    'L1_dropout'    : 0.2,

    # layer 2
    'L2_units'      : 120,
    'L2_act'        : 'relu',
    'L2_kernel_init': 'normal',
    'L2_kernel_reg' : keras.regularizers.l1(0.01),
    'L2_bias_reg'   : keras.regularizers.l1(0.01),
    'L2_dropout'    : 0.1,

    # layer 3
    'L3_units'      : 20,
    'L3_act'        : 'relu',
    'L3_kernel_init': 'normal',
    'L3_kernel_reg' : keras.regularizers.l1_l2(0.01),
    'L3_bias_reg'   : keras.regularizers.l1_l2(0.01),
    'L3_dropout'    : 0.1,

    # layer 4
    'L4_units'      : 10,
    'L4_act'        : 'relu',
    'L4_kernel_init': 'normal',
    'L4_kernel_reg' : None,
    'L4_bias_reg'   : None,
    'L4_dropout'    : 0.0,

    # NOTE: last layer is defined in model definition.

    # optimizer
    'optimizer': keras.optimizers.Nadam(learning_rate=0.001,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=1e-07,
                                        name="Nadam"),
}

#===============================================================================
METRICS = ['mae' ] # for regression val_mae gave better than val_mse

#===============================================================================
PARAMS_FIT = {'epochs': 500,
            'batch_size': 128,
            'patience': 20,
            'shuffle': True,
            'validation_split': 0.2
            }
