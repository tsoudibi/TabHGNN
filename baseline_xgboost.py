from data.dataset import *
from data.preprocess import *

from utils.utils import *
import yaml

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
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
    os.environ["WANDB_SILENT"] = "true"
    
    # slice K_fold
    
    # select dataset 
    # ['adult','compas','compass_old','eye_movements_num','eye_movements_cat','credit','electricity_cat']
    DATASET = 'adult'
    print('=================={}=================='.format(DATASET))
    # main_df = pd.read_csv(select_dataset(run_config['dataset']))
    main_df = pd.read_csv(select_dataset(DATASET))
    # main_df = randomize_df(main_df)
    # main_df = POOL_preprocess(main_df)
    TARGET = get_label_colunm()
    kf = KFold(n_splits=5, shuffle=True)
    AUCS = []
    ACCS = []
    # 進行5-fold交叉驗證
    for index, (train_index, test_index) in enumerate(kf.split(main_df)):
        X_train, X_test = main_df.loc[train_index], main_df.loc[test_index]
        '''===========CONFIGS==========='''
        K_BINS = False
        # MODEL = 'xgb'
        MODEL = 'rf'
        '''===========CONFIGS==========='''
        if K_BINS:
            X_train, inference_package,_,_ = POOL_preprocess(X_train, N_BINS = 100)
            X_test, _, _ = POOL_preprocess_inference(X_test, inference_package)
            # print(X_train)
            Y_train, Y_test = X_train[TARGET] - X_train[TARGET].min(), X_test[TARGET] - X_test[TARGET].min()
            X_train, X_test = X_train.drop(columns=TARGET), X_test.drop(columns=TARGET)
        else:
            from sklearn.preprocessing import LabelEncoder
            NUM, CAT, TARGET = get_feilds_attributes()
            for column in NUM:
                # min-max normalization
                #X_train[column] = (X_train[column] - X_train[column].mean()) / (X_train[column].std())
                #X_test[column] = (X_test[column] - X_test[column].mean()) / (X_test[column].std())
                X_train[column] = X_train[column].astype(np.float32)
                pass
            le = LabelEncoder()
            Y_train, Y_test = X_train[TARGET] , X_test[TARGET]
            Y_train = le.fit_transform(Y_train)
            Y_test = le.fit_transform(Y_test)

            
            X_train = X_train.drop(columns=TARGET)
            X_test = X_test.drop(columns=TARGET)
            
            for column in CAT:
                if column == TARGET:
                    continue
                le = LabelEncoder()
                le.fit(main_df[column].astype(str))
                X_train[column] = le.transform(X_train[column].astype(str))
                X_test[column] = le.transform(X_test[column].astype(str))
        # print(f'=========Fold {index}:==========')
        # print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
        # print(f'Y_train: {Y_train.shape}, Y_test: {Y_test.shape}')
        
        if MODEL == 'xgb':
            model = xgb.XGBClassifier(random_state=42, n_estimators =200)
        elif MODEL == 'rf':
            model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=2000)
        model.fit(X_train, Y_train)
        Y_prob = model.predict_proba(X_test)  # 预测概率
        acc = accuracy_score(Y_test, model.predict(X_test))
        auc = roc_auc_score(Y_test, Y_prob[:, 1])
        AUCS.append(auc)
        ACCS.append(acc)
        # print(f'auc: {auc}')
        # print(f'acc: {acc}')
    print("=================={}-{}==================".format(MODEL, K_BINS))
    print(f'Average AUC: {np.mean(AUCS)}, std: {np.std(AUCS)}')
    print(f'Average ACC: {np.mean(ACCS)}, std: {np.std(ACCS)}')