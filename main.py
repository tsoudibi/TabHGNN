from data.dataset import *
from data.preprocess import *

from model.model import *
from train.train import *
from utils.utils import *

from sklearn.model_selection import KFold
import wandb
import os
if __name__ == '__main__':    
    set_DEVICE('cuda')
    USE_WNADB = False

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    os.environ["WANDB_SILENT"] = "true"

    main_df = pd.read_csv('data/RAW_data/adult.csv')
    DEVICE = 'cuda'
    print('eweqr qw',get_DEVICE())
    # 進行5-fold交叉驗證
    for index, (train_index, test_index) in enumerate(kf.split(main_df)):
        if index !=0:
            continue
        print('[', index+1, 'fold] processing...')
        train_pool, test_pool = main_df.iloc[train_index], main_df.iloc[test_index]
        os.environ["WANDB_SILENT"] = "true"
        config = {
            "project": "K_fold test 0727 ",
            "name" : "v9.3 1000 " + str(index+1),
            "Max_epoch" : 1000,
            "group": "v9.3 1000  ",
            "learning_rate": 0.001,
            "batch_size": 128,
            "batch_size_test": 12,
            "K" : 100,
            "num_layers": 1,
            "embedding_dim": 128,
            "propagation_steps": 1,
            "unseen_rate": 0.0,
            "aug_strategy": "None",
            "N_BINS": 100,
            "random_state": 42,
            "notes": 'V10',
        }
        set_seed(config['random_state'])
        
        if USE_WNADB:
            wandb.init(
            # set the wandb project where this run will be logged
                project = config['project'], 
                name = config['name'],
                notes = config['notes'],
                entity = 'tabhyperformer',
                group = config['group'],
                # track hyperparameters and run metadata
                config = config
            )


        Main_data = HGNN_dataset( train_pool, 
                                label_column = 'income', 
                                test_df = test_pool, 
                                embedding_dim = config['embedding_dim'],
                                N_BINS = config['N_BINS'],
                                )
        model = TransformerDecoderModel(Main_data, 
                                        config['num_layers'], 
                                        config['embedding_dim'],
                                        config['propagation_steps']
                                        ).to(DEVICE)
        train(model, Main_data,
            epochs= config['Max_epoch'],
            lr = config['learning_rate'],
            batch_size = config['batch_size'],
            batch_size_test = config['batch_size_test'],
            K = config['K'],
            unseen_rate = config['unseen_rate'],
            aug_strategy = config['aug_strategy'],
            verbose = 2,
            log_name = config['name'],
            wandb_log = USE_WNADB)
        del Main_data, model
        
        if USE_WNADB:
            wandb.finish()