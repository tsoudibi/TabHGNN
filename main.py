from data.dataset import *
from data.preprocess import *

from model.model import *
from train.train import *
from utils.utils import *
import yaml

from sklearn.model_selection import KFold
import wandb
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
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # select dataset 
    main_df = pd.read_csv(select_dataset(run_config['dataset']))

    for index, (train_index, test_index) in enumerate(kf.split(main_df)):
        if index !=0:
            continue
        print('-----------------[', index+1, 'fold] processing...-----------------')
        train_pool, test_pool = main_df.iloc[train_index], main_df.iloc[test_index]

        set_seed(run_config['random_state'])
        
        if wandb_config['use_wandb']:
            wandb.init(
            # set the wandb project where this run will be logged
                project = wandb_config['project'], 
                name = wandb_config['name'],
                notes = wandb_config['notes'],
                entity = wandb_config['entity'],
                group = wandb_config['group'],
                # track hyperparameters and run metadata
                config = dict(run_config, **wandb_config)
            )


        Main_data = HGNN_dataset( train_pool, 
                                label_column = get_label_colunm(), 
                                test_df = test_pool, 
                                embedding_dim = run_config['embedding_dim'],
                                N_BINS = run_config['N_BINS'],
                                )
        model = TransformerDecoderModel(Main_data, 
                                        run_config['num_layers'], 
                                        run_config['embedding_dim'],
                                        run_config['propagation_steps']
                                        ).to(DEVICE)
        train(model, Main_data,
            epochs= run_config['max_epoch'],
            lr = run_config['learning_rate'],
            batch_size = run_config['batch_size'],
            batch_size_test = run_config['batch_size_test'],
            K = run_config['K'],
            unseen_rate = run_config['unseen_rate'],
            aug_strategy = run_config['aug_strategy'],
            evaluate_stride = run_config['evaluate_stride'],
            verbose = 2,
            log_name = wandb_config['name'],
            wandb_log = wandb_config['use_wandb'],)
        del Main_data, model
        
        if wandb_config['use_wandb']:
            wandb.finish()