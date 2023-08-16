from data.dataset import *
from data.preprocess import *

from utils.utils import *
import yaml

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import os
if __name__ == '__main__': 
    with open('config.yaml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    # data_config
    run_config = config['run_config']
    wandb_config = config['wandb_config']
    
    # initalize setting
    set_DEVICE(run_config['device'])
    DEVICE = get_DEVICE()
    set_seed(run_config['random_state'])
    set_PRINT_TIME(run_config['print_time'])
    select_dataset(run_config['dataset'])
    os.environ["WANDB_SILENT"] = "true"
    
    # slice K_fold
    
    # select dataset 
    main_df = pd.read_csv(select_dataset(run_config['dataset']))
    TARGET = get_label_colunm()
    kf = KFold(n_splits=5, shuffle=True)
    AUCS = []
    # 進行5-fold交叉驗證
    for index, (train_index, test_index) in enumerate(kf.split(main_df)):
        X_train, X_test = main_df.loc[train_index], main_df.loc[test_index]
        Y_train, Y_test = X_train[TARGET] - X_train[TARGET].min(), X_test[TARGET] - X_test[TARGET].min()
        X_train, X_test = X_train.drop(columns=TARGET), X_test.drop(columns=TARGET)
        print(f'Fold {index}:')
        print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
        print(f'Y_train: {Y_train.shape}, Y_test: {Y_test.shape}')
        
        model = xgb.XGBClassifier()
        model.fit(X_train, Y_train)
        Y_prob = model.predict_proba(X_test)  # 预测概率
        auc = roc_auc_score(Y_test, Y_prob[:, 1])
        AUCS.append(auc)
        print(f'auc: {auc}')
    print(f'Average AUC: {np.mean(AUCS)}')