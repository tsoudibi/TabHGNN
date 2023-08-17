import time

DEVICE = 'None'
def set_DEVICE(device):
    global DEVICE 
    DEVICE = device
def get_DEVICE():
    return DEVICE

import yaml 
DATA_CONFIG = {}
dataset = 'None'
with open('./data/RAW_data/data_configs.yml', 'r') as stream:
    DATA_CONFIG = yaml.load(stream, Loader=yaml.Loader)
    if DATA_CONFIG[dataset]['NUM'] == None:
        DATA_CONFIG[dataset]['NUM'] = []
    if DATA_CONFIG[dataset]['CAT'] == None:
        DATA_CONFIG[dataset]['CAT'] = []
def get_DATA_CONFIG():
    return DATA_CONFIG

def select_dataset(name):
    if name in DATA_CONFIG.keys():
        global dataset
        dataset = name
        print('=================[dataset is set to', name,']=================')
    else:
        raise ValueError('ERROR: dataset name is not in config file')
    return DATA_CONFIG[name]['file_path']

def get_feilds_attributes():
    return DATA_CONFIG[dataset]['NUM'], DATA_CONFIG[dataset]['CAT'], DATA_CONFIG[dataset]['TARGET']
def get_Discretizer_attributes():
    return DATA_CONFIG[dataset]['quntile'], DATA_CONFIG[dataset]['uniform']
def get_label_colunm():
    return DATA_CONFIG[dataset]['TARGET']

SEED = 0
def set_seed(seed):
    global SEED
    SEED = seed
    try:
        import tensorflow as tf
        tf.random.set_random_seed(seed)
        print("Tensorflow seed set successfully")
    except Exception as e:
        print("Set seed failed,details are ", e)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print("Pytorch seed set successfully")
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)
    # cuda env
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    
def randomize_df(df):
    df = df.sample(frac=1, random_state = SEED).reset_index(drop=True)
    return df
    
def check_DataFrame_distribution(X_trans):
    columns_range = {}
    print('%15s' % '', '%6s' % 'min','%6s' % 'max', '%6s' % 'nunique')
    
    for column in X_trans.columns:
        print('%15s' % column, '%6s' % X_trans[column].min(),'%6s' % X_trans[column].max(), '%6s' % X_trans[column].nunique())
        columns_range[column] = {}

T = time.time()
PRINT_TIME = True
def set_PRINT_TIME(flag):
    global PRINT_TIME
    PRINT_TIME = flag
def print_checkpoint_time(name):
    if PRINT_TIME:
        global T
        print('{0: <10}'.format(name),time.time()-T)
        T = time.time()
