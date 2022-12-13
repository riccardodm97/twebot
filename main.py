from argparse import ArgumentParser
from datetime import datetime

import pytz
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

import src.globals as glob
import src.utils as utils
import wandb
from src.data import (MultiTweetAndMetadataDataManager,
                      SingleTweetAndMetadataDataManager,
                      SingleTweetDataManager, TweetAndAccountDataManager)
from src.models import (MultiTweetAndMetadata_model, SingleTweet_model,
                        SingleTweetAndMetadata_model, TweetAndAccount_model)
from src.process import process_dataset
from src.trainer import Trainer


def main(task : str, action : str, debug : bool) :

    print(f"Starting task {task}")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'running on {DEVICE}')

    if task == 'MetadataSingleTweet': 

        #hyperparameters
        NUM_ESTIMATORS = 100
        CLASS_WEIGHT = 'balanced'
        RND_STATE = 18

        kwargs = {
            'v2': {'normalize':False}
            }

        dataset_df = process_dataset('v2', kwargs)

        train = dataset_df[dataset_df['split'] == 'train'].reset_index(drop=True)
        val = dataset_df[dataset_df['split'] == 'val'].reset_index(drop=True)
        test = dataset_df[dataset_df['split'] == 'test'].reset_index(drop=True)

        X_train, y_train = train.drop(columns=["account_id", "tweet", "label", "split", "processed_tweet"], axis=1), train["label"]
        X_val, y_val = val.drop(columns=["account_id", "tweet", "label", "split", "processed_tweet"], axis=1), val["label"]
        X_test, y_test = test.drop(columns=["account_id", "tweet", "label", "split", "processed_tweet"], axis=1), test["label"]

        rf = RandomForestClassifier(n_estimators=NUM_ESTIMATORS, class_weight=CLASS_WEIGHT, random_state=RND_STATE)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:,1] # keep probabilities for the positive outcome only

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_fpr, p_tpr, _ = roc_curve(y_test, [0] * len(y_test))
        auc_score = roc_auc_score(y_test, y_prob)

        print('acc:', acc)
        print('precision:', precision)
        print('recall:', recall)
        print('f1 score:', f1score)
        print('auc score:', auc_score)

        plt.style.use('seaborn')
        plt.plot(fpr, tpr, linestyle='--',color='orange', label='MetadataSingleTweet')
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')

        plt.legend(loc='best')
        plt.show();

    elif task == 'MetadataMultiTweet': 

        #hyperparameters
        NUM_ESTIMATORS = 100
        CLASS_WEIGHT = 'balanced'
        RND_STATE = 18
        NUM_TW_FEATURES = 30               # how many tweet from the same user are exploited to compute metadata features
        NUM_TW_TXT = 10                    # how many tweet from the same user are used as text for the lstm model 

        dataset_df = process_dataset('v3', {'tw_for_features':NUM_TW_FEATURES,'tw_for_txt':NUM_TW_TXT, 'normalize': False})

        train = dataset_df[dataset_df['split'] == 'train'].reset_index(drop=True)
        val = dataset_df[dataset_df['split'] == 'val'].reset_index(drop=True)
        test = dataset_df[dataset_df['split'] == 'test'].reset_index(drop=True)

        X_train, y_train = train.drop(columns=["account_id", "tweet", "label", "split", "processed_tweet", "n_tweet", "n_processed_tweet"], axis=1), train["label"]
        X_val, y_val = val.drop(columns=["account_id", "tweet", "label", "split", "processed_tweet", "n_tweet", "n_processed_tweet"], axis=1), val["label"]
        X_test, y_test = test.drop(columns=["account_id", "tweet", "label", "split", "processed_tweet", "n_tweet", "n_processed_tweet"], axis=1), test["label"]

        rf = RandomForestClassifier(n_estimators=NUM_ESTIMATORS, class_weight=CLASS_WEIGHT, random_state=RND_STATE)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:,1] # keep probabilities for the positive outcome only

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_fpr, p_tpr, _ = roc_curve(y_test, [0] * len(y_test))
        auc_score = roc_auc_score(y_test, y_prob)

        print('acc:', acc)
        print('precision:', precision)
        print('recall:', recall)
        print('f1 score:', f1score)
        print('auc score:', auc_score)

        plt.style.use('seaborn')
        plt.plot(fpr, tpr, linestyle='--',color='orange', label='MetadataMultiTweet')
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')

        plt.legend(loc='best')
        plt.show();

    elif task == 'Account' : 

        #hyperparameters
        NUM_ESTIMATORS = 100
        CLASS_WEIGHT = 'balanced'
        RND_STATE = 18

        dataset_df = process_dataset('account')

        train = dataset_df[dataset_df['split'] == 'train'].reset_index(drop=True)
        val = dataset_df[dataset_df['split'] == 'val'].reset_index(drop=True)
        test = dataset_df[dataset_df['split'] == 'test'].reset_index(drop=True)

        X_train, y_train = train.drop(columns=["account_id", "label", "split"], axis=1), train["label"]
        X_val, y_val = val.drop(columns=["account_id", "label", "split"], axis=1), val["label"]
        X_test, y_test = test.drop(columns=["account_id", "label", "split"], axis=1), test["label"]

        rf = RandomForestClassifier(n_estimators=NUM_ESTIMATORS, class_weight=CLASS_WEIGHT, random_state=RND_STATE)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:,1] # keep probabilities for the positive outcome only

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_fpr, p_tpr, _ = roc_curve(y_test, [0] * len(y_test))
        auc_score = roc_auc_score(y_test, y_prob)

        print('acc:', acc)
        print('precision:', precision)
        print('recall:', recall)
        print('f1 score:', f1score)
        print('auc score:', auc_score)

        plt.style.use('seaborn')
        plt.plot(fpr, tpr, linestyle='--',color='orange', label='Account')
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')

        plt.legend(loc='best')
        plt.show();

    elif task == "AccountAndMetadataMultiTweet":

        #hyperparameters
        NUM_ESTIMATORS = 100
        CLASS_WEIGHT = 'balanced'
        RND_STATE = 18
        NUM_TW_FEATURES = 30               # how many tweet from the same user are exploited to compute metadata features
        NUM_TW_TXT = 10                    # how many tweet from the same user are used as text for the lstm model 

        kwargs = {
                'v3': {'tw_for_features':NUM_TW_FEATURES,'tw_for_txt':NUM_TW_TXT, 'normalize': False},
                'v5': {'normalize': False}
                }

        dataset_df = process_dataset('v5', kwargs)

        train = dataset_df[dataset_df['split'] == 'train'].reset_index(drop=True)
        val = dataset_df[dataset_df['split'] == 'val'].reset_index(drop=True)
        test = dataset_df[dataset_df['split'] == 'test'].reset_index(drop=True)

        X_train, y_train = train.drop(columns=["account_id", "tweet", "label", "split", "processed_tweet", "n_tweet", "n_processed_tweet"], axis=1), train["label"]
        X_val, y_val = val.drop(columns=["account_id", "tweet", "label", "split", "processed_tweet", "n_tweet", "n_processed_tweet"], axis=1), val["label"]
        X_test, y_test = test.drop(columns=["account_id", "tweet", "label", "split", "processed_tweet", "n_tweet", "n_processed_tweet"], axis=1), test["label"]

        rf = RandomForestClassifier(n_estimators=NUM_ESTIMATORS, class_weight=CLASS_WEIGHT, random_state=RND_STATE)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:,1] # keep probabilities for the positive outcome only

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        p_fpr, p_tpr, _ = roc_curve(y_test, [0] * len(y_test))
        auc_score = roc_auc_score(y_test, y_prob)

        print('acc:', acc)
        print('precision:', precision)
        print('recall:', recall)
        print('f1 score:', f1score)
        print('auc score:', auc_score)

        plt.style.use('seaborn')
        plt.plot(fpr, tpr, linestyle='--',color='orange', label='AccountAndMetadataMultiTweet')
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')

        plt.legend(loc='best')
        plt.show();

    else : 

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

            dataset_df = process_dataset('v1')
            
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

            kwargs = {
            'v2': {'normalize':True}
            }

            dataset_df = process_dataset('v2', kwargs)
            
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


        elif task == 'MultiTweetAndMetadata':

            #hyperparameters
            BATCH_SIZE = 128                   # number of sentences in each mini-batch
            LR = 1e-3                          # learning rate 
            NUM_EPOCHS = 10                    # number of epochs
            WEIGHT_DECAY = 1e-4                # regularization
            LSTM_HIDDEN_DIM = 300              # hidden dimension of lstm network 
            LSTM_NUM_LAYERS = 1                # num of recurrent layers of lstm network 
            FREEZE = False                     # wheter to make the embedding layer trainable or not              
            DROPOUT = True                     # wheter to use dropout layer or not  
            DROPOUT_P = 0.5                    # dropout probability
            EMBEDDING_MODEL_NAME = 'fastText'  # which embedding model to use 
            NUM_TW_FEATURES = 30               # how many tweet from the same user are exploited to compute metadata features
            NUM_TW_TXT = 10                    # how many tweet from the same user are used as text for the lstm model 

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
                'emb_model_name' : EMBEDDING_MODEL_NAME,
                'num_tw_features' : NUM_TW_FEATURES,
                'num_tw_txt' : NUM_TW_TXT
            }

            dataset_df = process_dataset('v3',{'tw_for_features':NUM_TW_FEATURES,'tw_for_txt':NUM_TW_TXT})
            
            emb_model = utils.load_emb_model(EMBEDDING_MODEL_NAME)

            data_manager = MultiTweetAndMetadataDataManager(dataset_df,DEVICE)
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

            model = MultiTweetAndMetadata_model(data_manager.emb_matrix,model_cfg,DEVICE)
        
        elif task == 'TweetAndAccount':

            #hyperparameters
            BATCH_SIZE = 128                   # number of sentences in each mini-batch
            LR = 1e-3                          # learning rate 
            NUM_EPOCHS = 5                     # number of epochs
            WEIGHT_DECAY = 1e-3                # regularization
            LSTM_HIDDEN_DIM = 300              # hidden dimension of lstm network 
            LSTM_NUM_LAYERS = 1                # num of recurrent layers of lstm network 
            FREEZE = False                     # wheter to make the embedding layer trainable or not              
            DROPOUT = True                     # wheter to use dropout layer or not  
            DROPOUT_P = 0.5                    # dropout probability
            EMBEDDING_MODEL_NAME = 'fastText'  # which embedding model to use 
            NUM_TW_FEATURES = 30               # how many tweet from the same user are exploited to compute metadata features
            NUM_TW_TXT = 10                    # how many tweet from the same user are used as text for the lstm model 

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
                'emb_model_name' : EMBEDDING_MODEL_NAME,
                'num_tw_features' : NUM_TW_FEATURES,
                'num_tw_txt' : NUM_TW_TXT
            }

            kwargs = {
                'v3': {'tw_for_features':NUM_TW_FEATURES,'tw_for_txt':NUM_TW_TXT},
                'v4': {'normalize':True}
                }

            dataset_df = process_dataset('v4', kwargs)
            
            emb_model = utils.load_emb_model(EMBEDDING_MODEL_NAME)

            data_manager = TweetAndAccountDataManager(dataset_df,DEVICE)
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
                'txt_features_dim' : data_manager.text_features_dim,
                'acc_features_dim' : data_manager.account_features_dim
            }

            model = TweetAndAccount_model(data_manager.emb_matrix,model_cfg,DEVICE) 
        
        
        weight_positive_class = utils.get_weight_pos_class(dataset_df, DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight_positive_class)    #Binary CrossEntropy Loss that accept raw input and apply internally the sigmoid 
        optimizer = optim.Adam(model.parameters(), lr=LR , weight_decay=WEIGHT_DECAY)   #L2 regularization #TODO AdamW

        train_loader = data_manager.getDataloader('train', BATCH_SIZE, True)
        val_loader = data_manager.getDataloader('val', BATCH_SIZE, True)
        test_loader = data_manager.getDataloader('test', BATCH_SIZE, True)


        name = datetime.now(tz = pytz.timezone('Europe/Rome')).strftime("%d/%m/%Y %H:%M:%S") 
        wandb_mode = 'disabled' #if debug else None 
        wandb.init(project="tweebot", entity="uniboland", name=name, config=config, mode=wandb_mode, tags=[task], dir=str(glob.BASE_PATH))

        trainer = Trainer(model, DEVICE, criterion, optimizer)
        if action == "Train": trainer.train_and_eval(train_loader, val_loader, NUM_EPOCHS)
        else: trainer.test(test_loader)

        wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-t","--task", dest="task",help="Task to perform", 
        choices=
            # Random Forest
            ["MetadataSingleTweet", 
            "MetadataMultiTweet", 
            "AccountAndMetadataMultiTweet", 
            "Account",
            # LSTM
            "SingleTweet",
            "SingleTweetAndMetadata",
            "MultiTweetAndMetadata",
            "TweetAndAccount"],
            required=True)

    parser.add_argument("-a", "--action", dest="action", help="Train or Test the model", choices=["Train", "Test"])
    parser.add_argument("--debug",dest="debug",help="wheter to log on wandb or not", action="store_true")   
    args = parser.parse_args()

    main(args.task, args.action, args.debug)  