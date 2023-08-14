

DEVICE = 'None'
def set_DEVICE(device):
    global DEVICE 
    DEVICE = device
def get_DEVICE():
    return DEVICE

def set_seed(seed):
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
    
def check_DataFrame_distribution(X_trans):
    columns_range = {}
    print('%15s' % '', '%6s' % 'min','%6s' % 'max', '%6s' % 'nunique')
    
    for column in X_trans.columns:
        print('%15s' % column, '%6s' % X_trans[column].min(),'%6s' % X_trans[column].max(), '%6s' % X_trans[column].nunique())
        columns_range[column] = {}