from collections import OrderedDict, namedtuple
import json 

from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn
import torch 
import numpy as np 
import pandas as pd 

import src.globals as glob


Vocab = namedtuple('Vocabulary',['word2int','int2word','unique_words'])

class BaseDataManager():

    def __init__(self, dataframe : pd.DataFrame, device):

        self.device = device 
        self.dataset_df = dataframe.copy(deep=True)
    
    def custom_collate(self, batch):
        return NotImplementedError()

    def numericalize(self, token_list):  

        assert self.vocab is not None, "you have to build the vocab first, call build_vocab method to do it"
        return torch.tensor(list(map(self.vocab.word2int.get,token_list)))
    
    def build_vocab(self): 
        print('Building vocab...')

        unique_words : list = self.dataset_df['processed_tweet'].explode().unique().tolist()
        unique_words.insert(0,'<pad>')

        word2int = OrderedDict()
        int2word = OrderedDict()

        for i, word in enumerate(unique_words):
            word2int[word] = i           
            int2word[i] = word
        
        self.vocab = Vocab(word2int,int2word,unique_words)

        print(f'the number of unique words is {len(unique_words)}')
    
    def build_emb_matrix(self, emb_model): 
        print('Building embedding matrix...')

        embedding_dimension = emb_model.vector_size #how many numbers each emb vector is composed of                                                           
        embedding_matrix = np.zeros((len(self.vocab.word2int)+1, embedding_dimension), dtype=np.float32)   #create a matrix initialized with all zeros 

        for word, idx in self.vocab.word2int.items():
            if idx == 0: continue
            try:
                embedding_vector = emb_model[word]
            except (KeyError, TypeError):
                embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)

            embedding_matrix[idx] = embedding_vector     #assign the retrived or the generated vector to the corresponding index 
        
        self.emb_matrix = embedding_matrix
        
        print(f"Embedding matrix shape: {embedding_matrix.shape}")
    
    def getDataloader(self, split : str, batch_size : int, shuffle : bool):

        dataset = getattr(self,split+'_ds') 
        return DataLoader(dataset,batch_size,shuffle=shuffle,collate_fn=self.custom_collate)

class TweetDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame):
        self.tweet = dataframe['processed_tweet']
        self.label = dataframe['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'tweet': self.tweet[idx],
            'label': self.label[idx],
            }

class SingleTweetDataManager(BaseDataManager):

    def __init__(self, dataframe : pd.DataFrame, device):

        super().__init__(dataframe, device)

        self.train_ds = TweetDataset(self.dataset_df[self.dataset_df['split'] == 'train'].reset_index(drop=True))
        self.val_ds = TweetDataset(self.dataset_df[self.dataset_df['split'] == 'val'].reset_index(drop=True))
        self.test_ds = TweetDataset(self.dataset_df[self.dataset_df['split'] == 'test'].reset_index(drop=True))

    def custom_collate(self, batch):
        
        tweet_lengths = torch.tensor([len(example['tweet']) for example in batch]) #, device=self.device -> for pack_padded should be on cpu so if only used by that don't put it on gpu

        numerized_tweets = [self.numericalize(example['tweet']) for example in batch]
        padded_tweets = rnn.pad_sequence(numerized_tweets, batch_first = True, padding_value = self.vocab.word2int['<pad>']).to(self.device)

        labels = torch.tensor([example['label'] for example in batch],device=self.device) #(5)

        return {
            'tweets': padded_tweets,
            'labels': labels,
            'lengths': tweet_lengths
        }

class TweetAndMetadataDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame):
        self.tweet = dataframe['processed_tweet']
        self.label = dataframe['label']
        self.features = dataframe['features']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'tweet': self.tweet[idx],
            'label': self.label[idx],
            'features': self.features[idx]
            }

class TweetAndMultiFeaturesDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame):
        self.tweet = dataframe['processed_tweet']
        self.label = dataframe['label']
        self.txt_features = dataframe['text_features']
        self.acc_features = dataframe['account_features']


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'tweet': self.tweet[idx],
            'label': self.label[idx],
            'txt_features': self.txt_features[idx],
            'acc_features': self.acc_features[idx]
            }

class SingleTweetAndMetadataDataManager(BaseDataManager):  

    feature_columns = ['is_rt','url_c','tag_c','hashtag_c','cashtag_c','money_c','email_c','emoji_c','emoticon_c','len_tweet']

    def __init__(self, dataframe : pd.DataFrame, device):
        super().__init__(dataframe, device)

        self.dataset_df['features'] = self.dataset_df[self.feature_columns].values.tolist()
        self.metadata_features_dim = len(self.feature_columns)

        self.train_ds = TweetAndMetadataDataset(self.dataset_df[self.dataset_df['split'] == 'train'].reset_index(drop=True))
        self.val_ds = TweetAndMetadataDataset(self.dataset_df[self.dataset_df['split'] == 'val'].reset_index(drop=True))
        self.test_ds = TweetAndMetadataDataset(self.dataset_df[self.dataset_df['split'] == 'test'].reset_index(drop=True))
    
    def custom_collate(self, batch):
        
        tweet_lengths = torch.tensor([len(example['tweet']) for example in batch]) #, device=self.device -> for pack_padded should be on cpu so if only used by that don't put it on gpu

        features = torch.tensor([example['features'] for example in batch], device=self.device) 

        numerized_tweets = [self.numericalize(example['tweet']) for example in batch]
        padded_tweets = rnn.pad_sequence(numerized_tweets, batch_first = True, padding_value = self.vocab.word2int['<pad>']).to(self.device)

        labels = torch.tensor([example['label'] for example in batch],device=self.device) #(5)

        return {
            'tweets': padded_tweets,
            'features' : features,
            'labels': labels,
            'lengths': tweet_lengths
        }

class MultiTweetAndMetadataDataManager(SingleTweetAndMetadataDataManager):
    
    feature_columns = ['avg_length', 'avg_cleaned_length', '1+_mention', '1+_emot', '1+_url', 'max_hashtag', 'max_mention', 
    'unique_words_ratio', 'url_count', 'hashtag_count', 'unique_hashtag_ratio', 'mention_count', 'emot_count', 'punct_count', 
    'rt_count', 'unique_rt_ratio']

    def __init__(self, dataframe: pd.DataFrame, device):
        super().__init__(dataframe, device)


class TweetAndAccountDataManager(BaseDataManager):  

    text_features = ['avg_length','avg_cleaned_length','1+_mention','1+_emot','1+_url','max_hashtag','max_mentions','url_count','hashtag_count','mention_count',
    'emot_count','punct_count','?!_count','uppercased_count','cash_money_count','rt_count','unique_hashtag_ratio','unique_mention_ratio','unique_rt_ratio','unique_words_ratio']

    account_features = ['has_location', 'has_url', 'name_len', 'screen_name_len', 'description_len', 'followings_count','followers_count',
    'fofo_ratio', 'tweets_count', 'listed_count', 'num_in_screen_name', 'hashtag_in_description', 'def_profile', 'screen_name_entropy', 'tweet_freq', 'is_geo_enabled',
    'favourites_count', 'lev_dist_name_screeName', 'followers_growth_rate', 'followings_growth_rate', 'favourites_growth_rate']


    def __init__(self, dataframe : pd.DataFrame, device):
        super().__init__(dataframe, device)

        self.dataset_df['text_features'] = self.dataset_df[self.text_features].values.tolist()
        self.dataset_df['account_features'] = self.dataset_df[self.account_features].values.tolist()
        self.text_features_dim = len(self.text_features)
        self.account_features_dim = len(self.account_features)

        self.train_ds = TweetAndMultiFeaturesDataset(self.dataset_df[self.dataset_df['split'] == 'train'].reset_index(drop=True))
        self.val_ds = TweetAndMultiFeaturesDataset(self.dataset_df[self.dataset_df['split'] == 'val'].reset_index(drop=True))
        self.test_ds = TweetAndMultiFeaturesDataset(self.dataset_df[self.dataset_df['split'] == 'test'].reset_index(drop=True))
    
    def custom_collate(self, batch):
        
        tweet_lengths = torch.tensor([len(example['tweet']) for example in batch]) #, device=self.device -> for pack_padded should be on cpu so if only used by that don't put it on gpu

        txt_features = torch.tensor([example['txt_features'] for example in batch], device=self.device)
        acc_features = torch.tensor([example['acc_features'] for example in batch], device=self.device) 

        numerized_tweets = [self.numericalize(example['tweet']) for example in batch]
        padded_tweets = rnn.pad_sequence(numerized_tweets, batch_first = True, padding_value = self.vocab.word2int['<pad>']).to(self.device)

        labels = torch.tensor([example['label'] for example in batch],device=self.device) #(5)

        return {
            'tweets': padded_tweets,
            'txt_features' : txt_features,
            'acc_features' : acc_features,
            'labels': labels,
            'lengths': tweet_lengths
        }



def loadData():

    json_file_path_train = glob.DATA_FOLDER / 'Twibot-20/train.json'
    json_file_path_val = glob.DATA_FOLDER / 'Twibot-20/dev.json'
    json_file_path_test = glob.DATA_FOLDER / 'Twibot-20/test.json'

    with open(json_file_path_train, 'r') as tr:
        contents = json.loads(tr.read())
        train_df = pd.json_normalize(contents)
        train_df['split'] = 'train'

    with open(json_file_path_val, 'r') as vl:
        contents = json.loads(vl.read())
        val_df = pd.json_normalize(contents) 
        val_df['split'] = 'val'

    with open(json_file_path_test, 'r') as ts:
        contents = json.loads(ts.read())
        test_df = pd.json_normalize(contents) 
        test_df['split'] = 'test'

    df = pd.concat([train_df,val_df,test_df],ignore_index=True) # merge three datasets
    df.dropna(subset=['tweet'], inplace=True)  # remove rows withot any tweet 
    df.set_index(keys='ID',inplace=True) # reset index

    # split dataframe in two : tweet and account data 
    tweets_df = df[['tweet','label','split']].reset_index()
    tweets_df = tweets_df.explode('tweet').reset_index(drop=True)
    tweets_df.rename(columns={"ID": "account_id"}, inplace=True)

    account_df = df.drop('tweet',axis=1).reset_index()
    account_df.rename(columns={"ID": "account_id"}, inplace=True)

    return tweets_df, account_df




