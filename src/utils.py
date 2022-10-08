import random 
import os 

import gensim
import gensim.downloader as gloader
from gensim.models import KeyedVectors
import gdown 
import pandas as pd 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch 

import src.globals as glob 


def load_emb_model(name : str, force_download : bool = False) :

    assert name == 'fastText' or name == 'glove', 'the two embedding models available are glove and fasttext'

    emb_model_cached_path = "twitter-multilingual-300d.new.bin" if name == 'fastText' else 'glove-twitter-200.bin'   
    emb_model_cached_path = glob.DATA_FOLDER / emb_model_cached_path

    if not os.path.exists(emb_model_cached_path) or force_download: 
        print('downloading embedding model...')    

        if name == 'fastText':   
            gdown.download(id="1DprdHGocFXJ9swnb2pDJJxHw5QR810LS",output=str(emb_model_cached_path))
        
        else :
            model : KeyedVectors = gloader.load('glove-twitter-200')
            model.save_word2vec_format(emb_model_cached_path, binary=True)   
    else : 
        print('found cached emb_model in data folder, retrieving the file...')

    emb_model = KeyedVectors.load_word2vec_format(emb_model_cached_path, binary=True)
    print('vectors loaded')

    return emb_model


def check_OOV_terms(embedding_model: gensim.models.keyedvectors.KeyedVectors, unique_words):
    """
        Given the embedding model and the unique words in the dataframe, determines the out-of-vocabulary words 
    """
    oov_words = []

    if embedding_model is None:
        print('WARNING: empty embeddings model')

    else: 
        for word in unique_words:
            try: 
                embedding_model[word]
            except:
                oov_words.append(word) 
        
        print("Number of unique words in dataset:",len(unique_words))
        print(f"Total OOV terms: {len(oov_words)} which is ({(float(len(oov_words)) / len(unique_words))*100:.2f}%)")
        print("Some OOV terms:",random.sample(oov_words,10))
    
    return oov_words


def metrics(y_true, y_pred):
    """
        Compute accuracy and f1-score for an epoch 
    """
    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true,y_pred,average='macro')

    prec = precision_score(y_true,y_pred,average='macro')

    rec = recall_score(y_true,y_pred,average="macro")

    return acc, f1, prec, rec


def get_weight_pos_class(dataframe : pd.DataFrame, device) :

    #to counteract class imbalance 
    train = dataframe[dataframe['split']=='train']
    counts = train['label'].value_counts().to_dict()
    (human, bot) = counts[0.0], counts[1.0]
    weight_positive_class = torch.tensor([human/bot], device = device)  #weight to give to positive class 

    return weight_positive_class
    

def check_correlation(dataframe : pd.DataFrame, column_names : list[str], target_column : str):

    print(dataframe[column_names].corrwith(dataframe[target_column]))



