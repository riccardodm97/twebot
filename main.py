
from argparse import ArgumentParser
from datetime import datetime
import pytz
import os 

import torch 
import torch.nn as nn
import torch.optim as optim
import wandb 
import pandas as pd 

from src.data import SingleTweetAndMetadataDataManager, SingleTweetDataManager, loadData
from src.models import SingleTweet_model, SingleTweetAndMetadata_model
import src.utils as utils
from src.trainer import Trainer
import src.globals as glob
from src.process import process_dataset_v1, process_dataset_v2


def main(task : str, debug : bool) :

    print(f"Starting task {task}")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on {DEVICE}')


    if task == 'SingleTweet':

        #hyperparameters
        BATCH_SIZE = 512                   # number of sentences in each mini-batch
        LR = 1e-3                          # learning rate 
        NUM_EPOCHS = 5                     # number of epochs
        WEIGHT_DECAY = 1e-5                # regularization
        LSTM_HIDDEN_DIM = 300              # hidden dimension of lstm network 
        LSTM_NUM_LAYERS = 1                # num of recurrent layers of lstm network 
        FREEZE = False                     # wheter to make the embedding layer trainable or not              
        DROPOUT = True                     # wheter to use dropout layer or not  
        DROPOUT_P = 0.5                    # dropout probability
        EMBEDDING_MODEL_NAME = 'fastText'  # which embedding model to use 

        config = {
            'batch_size' : BATCH_SIZE,
            'lr' : LR,
            'num_epochs' : NUM_EPOCHS,
            'weight_decay' : WEIGHT_DECAY,
            'lstm_hidden_dim' : LSTM_HIDDEN_DIM,
            'lstm_num_layers': LSTM_NUM_LAYERS,
            'freeze' : FREEZE,
            'dropout' : DROPOUT,
            'dropout_p' : DROPOUT_P,
            'device' : DEVICE,
            'emb_model_name': EMBEDDING_MODEL_NAME
        }

        dataset_path = glob.DATA_FOLDER / 'processed_dataset_v1.pkl'

        if not os.path.exists(dataset_path) or glob.force_processing:
            tweets_df, account_df = loadData()
            dataset_df = process_dataset_v1(tweets_df,dataset_path)
        else : 
            print('found already processed dataset in data folder, retrieving the file...')
            dataset_df = pd.read_pickle(dataset_path)
            print('dataset loaded in Dataframe')
        
        emb_model = utils.load_emb_model(EMBEDDING_MODEL_NAME)

        data_manager = SingleTweetDataManager(dataset_df,DEVICE)
        data_manager.build_vocab()
        data_manager.build_emb_matrix(emb_model)

        # model config parameters dictionary
        model_cfg = {
            'pad_idx' : data_manager.vocab.word2int['<pad>'],
            'freeze_embedding' : FREEZE,  
            'dropout' : DROPOUT,
            'dropout_p' : DROPOUT_P,
            'hidden_dim' : LSTM_HIDDEN_DIM,
            'num_layers': LSTM_NUM_LAYERS
        }

        model = SingleTweet_model(data_manager.emb_matrix,model_cfg,DEVICE)


    elif task == 'SingleTweetAndMetadata':

        #hyperparameters
        BATCH_SIZE = 512                   # number of sentences in each mini-batch
        LR = 1e-3                          # learning rate 
        NUM_EPOCHS = 5                     # number of epochs
        WEIGHT_DECAY = 1e-3                # regularization
        LSTM_HIDDEN_DIM = 300              # hidden dimension of lstm network 
        LSTM_NUM_LAYERS = 1                # num of recurrent layers of lstm network 
        FREEZE = False                     # wheter to make the embedding layer trainable or not              
        DROPOUT = True                     # wheter to use dropout layer or not  
        DROPOUT_P = 0.5                    # dropout probability
        EMBEDDING_MODEL_NAME = 'glove'  # which embedding model to use 

        config = {
            'batch_size' : BATCH_SIZE,
            'lr' : LR,
            'num_epochs' : NUM_EPOCHS,
            'weight_decay' : WEIGHT_DECAY,
            'lstm_hidden_dim' : LSTM_HIDDEN_DIM,
            'lstm_num_layers': LSTM_NUM_LAYERS,
            'freeze' : FREEZE,
            'dropout' : DROPOUT,
            'dropout_p' : DROPOUT_P,
            'device' : DEVICE,
            'emb_model_name': EMBEDDING_MODEL_NAME
        }

        dataset_path_v1 = glob.DATA_FOLDER / 'processed_dataset_v1.pkl'
        dataset_path_v2 = glob.DATA_FOLDER / 'processed_dataset_v2.pkl'

        if not os.path.exists(dataset_path_v2) or glob.force_processing:

            if not os.path.exists(dataset_path_v1) or glob.force_processing:
                tweets_df, account_df = loadData()
                dataset_df = process_dataset_v1(tweets_df,dataset_path_v1)
            else : 
                dataset_df = pd.read_pickle(dataset_path_v1)
            
            dataset_df = process_dataset_v2(dataset_df,dataset_path_v2)
        else : 
            print('found already processed dataset in data folder, retrieving the file...')
            dataset_df = pd.read_pickle(dataset_path_v2)
            print('dataset loaded in Dataframe')
        
        emb_model = utils.load_emb_model(EMBEDDING_MODEL_NAME)

        data_manager = SingleTweetAndMetadataDataManager(dataset_df,DEVICE)
        data_manager.build_vocab()
        data_manager.build_emb_matrix(emb_model)

        # model config parameters dictionary
        model_cfg = {
            'pad_idx' : data_manager.vocab.word2int['<pad>'],
            'freeze_embedding' : FREEZE,  
            'dropout' : DROPOUT,
            'dropout_p' : DROPOUT_P,
            'hidden_dim' : LSTM_HIDDEN_DIM,
            'num_layers': LSTM_NUM_LAYERS,
            'metadata_features_dim' : data_manager.metadata_features_dim
        }

        model = SingleTweetAndMetadata_model(data_manager.emb_matrix,model_cfg,DEVICE)

    
    weight_positive_class = utils.get_weight_pos_class(dataset_df, DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_positive_class)    #Binary CrossEntropy Loss that accept raw input and apply internally the sigmoid 
    optimizer = optim.Adam(model.parameters(), lr=LR , weight_decay=WEIGHT_DECAY)   #L2 regularization 

    train_loader = data_manager.getDataloader('train', BATCH_SIZE, True)
    val_loader = data_manager.getDataloader('val', BATCH_SIZE, True)


    name = datetime.now(tz = pytz.timezone('Europe/Rome')).strftime("%d/%m/%Y %H:%M:%S") 
    wandb_mode = 'disabled' if debug else None 
    wandb.init(project="tweebot", entity="uniboland", name=name, config=config, mode=wandb_mode, tags=[task], dir=str(glob.BASE_PATH))

    trainer = Trainer(model, DEVICE, criterion, optimizer)
    trainer.train_and_eval(train_loader, val_loader, NUM_EPOCHS)

    wandb.finish()



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-t","--task", dest="task",help="Task to perform", choices=["SingleTweet","SingleTweetAndMetadata"], required=True)
    parser.add_argument("--debug",dest="debug",help="wheter to log on wandb or not", action="store_true")   
    args = parser.parse_args()

    main(args.task,args.debug)  