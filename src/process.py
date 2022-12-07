import collections
import math
import os
import re
from datetime import datetime
from statistics import mean
from string import punctuation

import emoji
import Levenshtein
import nltk
import pandas as pd
from emot.core import emot
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from pandas.core.common import flatten
from sklearn.preprocessing import StandardScaler
from ttp import ttp

import src.globals as glob
from src.data import loadData

nltk.download('stopwords',glob.DATA_FOLDER)


def process_dataset_v1(dataframe : pd.DataFrame , save_path ) :

    parser = ttp.Parser(include_spans=True)
    emot_obj = emot()
    tk = TweetTokenizer(reduce_len=True,preserve_case=False)

    RETWEET = r"^RT (?:@[\w_]+):"
    NEWLINE = r"\n"
    CASHTAG = r"(?<!\S)\$[A-Z]+(?:\.[A-Z]+)?(?!\S)"
    EMAIL = r"""[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]"""
    MONEY = r"[$£][0-9]+(?:[.,]\d+)?[Kk+BM]?|[0-9]+(?:[.,]\d+)?[Kk+BM]?[$£]"
    NUMBER = r"""(?<!\S)(?:[+\-]?\d+(?:%|(?:[,/.:-]\d+[+\-]?)?))"""
    HASHTAG = r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    HANDLE = r"""(?:@[\w_]+)"""

    TO_REPLACE = [RETWEET, NEWLINE, CASHTAG, EMAIL, MONEY, NUMBER, HASHTAG, HANDLE]
    REPLACE_WITH = [' retweet ',' ',' stock ',' email ',' money ',' number ',' hashtag ',' username ']


    def replace(word : str):
        if not word.isascii():
            return ['']
        if bool(re.search(r'http[s]?|.com',word)):
            return ['url']
        elif bool(re.search(r'\d',word)):
            return ['number']
        elif bool(re.search(r'haha|ahah|jaja|ajaj',word)):
            return ['ahah']
        elif bool(re.search(r'\n',word)):
            return ['']
        elif bool(re.search('-',word)):
            return re.sub('-',' ',word).split()
        elif bool(re.search("'",word)):
            return re.sub("'"," '",word).split()     #CHANGE ? 
        else :
            return [word] 
        

    def further_process(sentence: str):

        #replace urls 
        result = parser.parse(sentence, html=False)
        urls = dict(result.urls).keys()
        for url in urls:
            sentence = sentence.replace(url,' url ')
        
        #replace emoticons 
        emoticons = emot_obj.emoticons(sentence)
        for emoticon in emoticons['value']:
            sentence = sentence.replace(emoticon,' emoticon ')
        
        #replace emoji
        sentence = emoji.replace_emoji(sentence,' emoji ')

        #tokenize
        sentence = tk.tokenize(sentence)

        #replace residual wrong words 
        sentence = [w for word in sentence for w in replace(word)]
        
        #remove empty strings 
        sentence = [word for word in sentence if word != '']
                
        return sentence

    dataframe['processed_tweet'] = dataframe['tweet'].replace(TO_REPLACE,REPLACE_WITH,regex=True,inplace=False)
    dataframe['processed_tweet'] = dataframe['processed_tweet'].apply(further_process)

    dataframe = dataframe[dataframe['processed_tweet'].map(lambda x: len(x)) > 2].reset_index(drop=True)     #TODO leave it or not ?? 

    dataframe['label'] = dataframe['label'].astype(float)  

    print('saving processed dataset to file')
    dataframe.to_pickle(save_path)   #save to file 
    
    return dataframe
        


def process_dataset_v2(dataframe : pd.DataFrame, save_path, normalize : bool) :

    sw = stopwords.words('english')

    df = dataframe.copy(deep=True)  

    def is_retweet(sentence_list : list):
        return float(sentence_list[0] == 'retweet')

    def url_count(sentence_list : list):
        c = sentence_list.count('url')
        return c
        
    def tag_count(sentence_list : list):
        c = sentence_list.count('username')
        return c

    def hashtag_count(sentence_list : list):
        c = sentence_list.count('hashtag')
        return c

    def cashtag_count(sentence_list : list):
        c = sentence_list.count('stock')
        return c

    def money_count(sentence_list : list):
        c = sentence_list.count('money')
        return c

    def email_count(sentence_list : list):
        c = sentence_list.count('email')
        return c
        
    def number_count(sentence_list : list):
        c = sentence_list.count('number')
        return c

    def emoticon_count(sentence_list : list):
        c = sentence_list.count('emoticon')
        return c

    def emoji_count(sentence_list : list):
        c = sentence_list.count('emoji')
        return c

    def stopwords_count(sentence_list : list):
        c = 0
        for word in sentence_list : 
            if word in sw:
                c+=1
            
        return c 

    def punct_count(sentence_list : list):
        c = 0
        for word in sentence_list : 
            if word in punctuation:
                c+=1
            
        return c 

    df['is_rt'] = df['processed_tweet'].apply(is_retweet)
    df['url_c'] = df['processed_tweet'].apply(url_count)
    df['tag_c'] = df['processed_tweet'].apply(tag_count)
    df['hashtag_c'] = df['processed_tweet'].apply(hashtag_count)
    df['cashtag_c'] = df['processed_tweet'].apply(cashtag_count)
    df['money_c'] = df['processed_tweet'].apply(money_count)
    df['email_c'] = df['processed_tweet'].apply(email_count)
    df['number_c'] = df['processed_tweet'].apply(number_count)
    df['emoji_c'] = df['processed_tweet'].apply(emoji_count)
    df['emoticon_c'] = df['processed_tweet'].apply(emoticon_count)
    df['len_tweet'] = df['processed_tweet'].apply(len)
    df['stopwords_c'] = df['processed_tweet'].apply(stopwords_count)
    df['punct_c'] = df['processed_tweet'].apply(punct_count)

    to_ignore = ['number_c','stopwords_c','punct_c']
    df = df.drop(to_ignore,axis=1).reset_index(drop=True)  

    # column_names = ['url_c','tag_c','hashtag_c','cashtag_c','money_c','email_c','number_c','emoji_c','emoticon_c','len_tweet',
    # 'stopwords_c','punct_c']
    column_names = df.columns.difference(['tweet','account_id','label','split','processed_tweet','is_rt'])

    train_df = df[df['split'] =='train']
    if normalize == True:
        scaler = StandardScaler()
        scaler.fit(train_df[column_names])

        df[column_names] = scaler.transform(df[column_names])

    print('saving processed dataset to file')
    df.to_pickle(save_path)   #save to file

    return df 


def process_dataset_v3(dataframe : pd.DataFrame, save_path : str, tw_for_features : int, tw_for_txt : int, normalize : bool) :

    sw = stopwords.words('english')
    df = dataframe.copy(deep=True)
     
    
    def clean_tweet(tweet: list ):
        to_remove = ['retweet','username','hashtag','url','emoticon','emoji','number','stock','money','email']
        return [x for x in tweet if x not in to_remove and x not in punctuation and x not in sw]


    #BEFORE COLLAPSING ALL TWEET IN ONE

    def avg_tweet_length(sentence_list : list[list]):
        return mean([len(sentence) for sentence in sentence_list])

    def avg_cleaned_tweet_length(sentence_list : list[list]) :
        return mean([len(clean_tweet(sentence)) for sentence in sentence_list])

    def tweet_with_atleast_one_mention(sentence_list : list[list]):
        n = 0
        for sentence in sentence_list:
            if 'username' in sentence:
                n+=1
        
        return n

    def tweet_with_atleast_one_emot(sentence_list : list[list]):
        n = 0
        for sentence in sentence_list:
            if 'emoji' in sentence or 'emoticon' in sentence:
                n+=1
        
        return n

    def tweet_with_atleast_one_url(sentence_list : list[list]):
        n = 0
        for sentence in sentence_list:
            if 'url' in sentence:
                n+=1
        
        return n

    def max_hashtags_single_tweet(sentence_list : list[list]):
        return max([sentence.count('hashtag') for sentence in sentence_list])

    def max_mentions_single_tweet(sentence_list : list[list]):
        return max([sentence.count('username') for sentence in sentence_list])

    def unique_words_ratio(sentence_list : list[list]):
        s = []
        for sentence in sentence_list:
            if sentence[0] != 'retweet':
                s.extend(clean_tweet(sentence))
        
        if s : return len(set(s)) / len(s)
        else : return 1.0


    #AFTER COLLAPSING ALL TWEET IN ONE

    def URLs_count(proc_sentence : list):
        return proc_sentence.count('url')

    def hashtag_count(proc_sentence : list):
        return proc_sentence.count('hashtag')

    def unique_hashtag_ratio(sentence : str):

        tags = re.findall(r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",sentence)
        if tags :
            return len(set(tags)) / len(tags) 
        else : return 1.0 
        
    def mention_count(proc_sentence : list):
        return proc_sentence.count('username')

    def unique_mention_ratio(sentence : str):

        mentions = re.findall(r"""(?<!RT )(?:@[\w_]+)""",sentence)
        if mentions :
            return len(set(mentions)) / len(mentions) 
        else : return 1.0 

    def emoticon_emoji_count(proc_sentence : list):
        return proc_sentence.count('emoticon') + proc_sentence.count("emoji")

    def punctuation_count(proc_sentence : list):
        c = 0
        for word in proc_sentence : 
            if word in punctuation:
                c+=1
            
        return c 

    def question_and_exclamation_mark_count(proc_sentence : list):
        return proc_sentence.count('?') + proc_sentence.count("!")

    def uppercased_word_count(proc_sentence : list):
        return sum([str.isupper(word) for word in proc_sentence])

    def cashtag_money_count(proc_sentence : list):
        return proc_sentence.count('stock') + proc_sentence.count("money")   #TODO cosi si confondon anche con le parole stock e money originali 

    def retweet_count(proc_sentence : list):
        return proc_sentence.count('retweet')

    def unique_retweet_ratio(sentence : str):

        rt = re.findall(r"RT (?:@[\w_]+):",sentence)
        if rt :
            return len(set(rt)) / len(rt) 
        else : return 1.0 

    # AGGREGATE TWEET FROM SAME ACCOUNT 
    
    aggregation_functions = {'account_id': 'first', 'tweet': lambda x : x.tolist(), 'label': 'first', 'split': 'first','processed_tweet': lambda x : x.tolist()}
    df = df.groupby(df['account_id'],as_index=False,sort=False).agg(aggregation_functions) 
    df = df [df['tweet'].map(lambda x: len(x)) >= tw_for_features].reset_index(drop=True) 
    df['n_tweet'] = df['tweet'].map(lambda x: x[:tw_for_features])
    df['n_processed_tweet'] = df['processed_tweet'].map(lambda x: x[:tw_for_features])


    df['avg_length'] = df['n_processed_tweet'].apply(avg_tweet_length)
    df['avg_cleaned_length'] = df['n_processed_tweet'].apply(avg_cleaned_tweet_length)
    df['1+_mention'] = df['n_processed_tweet'].apply(tweet_with_atleast_one_mention)
    df['1+_emot'] = df['n_processed_tweet'].apply(tweet_with_atleast_one_emot)
    df['1+_url'] = df['n_processed_tweet'].apply(tweet_with_atleast_one_url)
    df['max_hashtag'] = df['n_processed_tweet'].apply(max_hashtags_single_tweet)
    df['max_mentions'] = df['n_processed_tweet'].apply(max_mentions_single_tweet)
    df['unique_words_ratio'] = df['n_processed_tweet'].apply(unique_words_ratio)

    # COLLAPSE MULTIPLE TWEET IN ONE 

    df['n_tweet'] = df['n_tweet'].apply(' '.join)
    df['n_processed_tweet'] = df['n_processed_tweet'].apply(lambda x : list(flatten(x)))

    df['url_count'] = df['n_processed_tweet'].apply(URLs_count)
    df['hashtag_count'] = df['n_processed_tweet'].apply(hashtag_count)
    df['unique_hashtag_ratio'] = df['n_tweet'].apply(unique_hashtag_ratio)
    df['mention_count'] = df['n_processed_tweet'].apply(mention_count)
    df['unique_mention_ratio'] = df['n_tweet'].apply(unique_mention_ratio)
    df['emot_count'] = df['n_processed_tweet'].apply(emoticon_emoji_count)
    df['punct_count'] = df['n_processed_tweet'].apply(punctuation_count)
    df['?!_count'] = df['n_processed_tweet'].apply(question_and_exclamation_mark_count)
    df['uppercased_count'] = df['n_processed_tweet'].apply(uppercased_word_count)
    df['cash_money_count'] = df['n_processed_tweet'].apply(cashtag_money_count)
    df['rt_count'] = df['n_processed_tweet'].apply(retweet_count)
    df['unique_rt_ratio'] = df['n_tweet'].apply(unique_retweet_ratio)

    
    # REPEAT THE PROCESS OF SELECTION AND COLLAPSING IN CASE WE WANT A DIFFERENT NUMBER OF TWEETS FOR TEXT THAN FOR FEATURES  
    df['tweet'] = df['tweet'].map(lambda x: x[:tw_for_txt])
    df['processed_tweet'] = df['processed_tweet'].map(lambda x: x[:tw_for_txt])
    df['tweet'] = df['tweet'].apply(' '.join)
    df['processed_tweet'] = df['processed_tweet'].apply(lambda x : list(flatten(x)))

    # REMOVE NON USEFUL FEATURES 

    to_ignore = ['?!_count','cash_money_count','unique_mention_ratio','uppercased_count']
    df = df.drop(to_ignore,axis=1).reset_index(drop=True)  

    # APPLY NORMALIZATION 

    column_names = df.columns.difference(['tweet','account_id','label','split','processed_tweet','n_tweet','n_processed_tweet'])
    # column_names = ['avg_length','avg_cleaned_length','1+_mention','1+_emot','1+_url','max_hashtag','max_mentions','url_count',
    # 'hashtag_count','mention_count','emot_count','punct_count','?!_count','uppercased_count','cash_money_count','rt_count']
    #column_names.extend(['unique_hashtag_ratio','unique_mention_ratio','unique_rt_ratio','unique_words_ratio'])  #TODO ??
    #df[column_names] = df[column_names].apply(zscore)

    train_df = df[df['split'] =='train']
    if normalize == True:
        scaler = StandardScaler()
        scaler.fit(train_df[column_names])

        df[column_names] = scaler.transform(df[column_names])

    print('saving processed dataset to file')
    df.to_pickle(save_path)   #save to file

    return df 


def process_account_dataset(dataframe : pd.DataFrame, normalize : bool) :

    to_drop = ['neighbor','domain','profile.id','profile.id_str','profile.profile_location','profile.entities','profile.utc_offset','profile.time_zone','profile.lang',
'profile.contributors_enabled','profile.is_translator','profile.is_translation_enabled','profile.profile_background_color','profile.profile_background_image_url','profile.profile_background_image_url_https',
'profile.profile_background_tile','profile.profile_image_url','profile.profile_image_url_https','profile.profile_link_color','profile.profile_sidebar_border_color','profile.profile_sidebar_fill_color',
'profile.profile_text_color','profile.has_extended_profile','neighbor.following', 'neighbor.follower']

    rename_dict = {'profile.name': 'name', 'profile.screen_name': 'screen_name', 'profile.location': 'location','profile.description':'description',
'profile.url' : 'url', 'profile.protected' : 'protected', 'profile.followers_count':'followers_count', 'profile.friends_count':'friends_count', 
'profile.listed_count':'listed_count','profile.created_at' :'created_at', 'profile.favourites_count' :'favourites_count','profile.geo_enabled':'geo_enabled',
'profile.verified':'verified', 'profile.statuses_count':'statuses_count','profile.profile_use_background_image' : 'background_image',
'profile.default_profile' : 'default_profile', 'profile.default_profile_image':'default_profile_image'}


    to_ignore = ['is_verified','has_desc','bot_word_in_name','bot_word_in_screen_name','bot_word_in_description','num_in_name','urls_in_description','def_image','is_protected','use_background_img']

    
    def description_len(desc:str):
        if desc == "":
            return 0
        else : 
            return len(desc)

    def fofo_ratio(f1: str ,f2: str):
        f1,f2 = int(f1), int(f2)
        if f2 == 0.0 : 
            return f1
        else : 
            return f1/f2 

    def numbers_in_str(s : str):
        return len(re.findall(r'\d+', s))
            
    def hashtag_in_str(s: str):
        return len(re.findall(r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",s))

    def urls_in_str(s: str):
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', s)
        return len(urls)

    def bot_world_in_str(s: str):
        return int(bool(re.search('bot', s, flags=re.IGNORECASE)))


    def get_str_entropy(s : str):
        counter = collections.Counter(s)
        screen_name_entropy = 0
        for key, cnt in counter.items():
            prob = float(cnt) / len(s)
            screen_name_entropy += -1 * prob * math.log(prob, 2)
        return screen_name_entropy
        
    def get_account_age(created_at : str) :
        Twibot_20_FORMAT = f'%a %b %d %H:%M:%S %z %Y'
        creation_date = datetime.strptime(created_at,Twibot_20_FORMAT).date()
        present_date = datetime.now().date()
        user_age = (present_date - creation_date).days
        return int(user_age)


    def frequency(created_at : str, numerator : str):
        age = get_account_age(created_at)
        avg_tweets = int(numerator) / age 
        return avg_tweets

    def lev_name_screenName(name : str, screen_name : str) :
        return Levenshtein.distance(name,screen_name)


    df = dataframe.copy(deep=True)

    df = df.drop(to_drop,axis=1).reset_index(drop=True)  # drop unuseful columns 
    df.rename(columns=rename_dict, inplace=True)         # rename all columns 
    df = df.applymap(lambda x : x.rstrip())              # strip the trailing space added by twitter 


    feature_df = df[['account_id','label','split']].reset_index(drop=True)
    feature_df['label'] = feature_df['label'].astype(float)


    #KOUVELA 
    feature_df['has_desc'] = (df['description'] != '').astype(int)  #check lenght 1486
    feature_df['has_location'] = (df['location'] != '').astype(int)  #check lenght 3386
    feature_df['has_url'] = (df['url'] != 'None').astype(int)  #check lenght 5965
    feature_df['is_verified'] = (df['verified'] == 'True').astype(int)  #check lenght 2962
    feature_df['bot_word_in_name'] = df['name'].apply(bot_world_in_str)
    feature_df['bot_word_in_screen_name'] = df['screen_name'].apply(bot_world_in_str)
    feature_df['bot_word_in_description'] = df['description'].apply(bot_world_in_str)
    feature_df['name_len'] = df['name'].apply(len)
    feature_df['screen_name_len'] = df['screen_name'].apply(len)
    feature_df['description_len'] = df['description'].apply(description_len)
    feature_df['followings_count'] = df['friends_count'].astype(int)
    feature_df['followers_count'] = df['followers_count'].astype(int)
    feature_df['fofo_ratio'] = df.apply(lambda x: fofo_ratio(x['followers_count'], x['friends_count']), axis=1)
    feature_df["tweets_count"] = df['statuses_count'].astype(int)
    feature_df["listed_count"] = df['listed_count'].astype(int)
    feature_df["num_in_name"] = df['name'].apply(numbers_in_str)
    feature_df["num_in_screen_name"] = df['screen_name'].apply(numbers_in_str)
    feature_df['hashtag_in_description'] = df['description'].apply(hashtag_in_str)
    feature_df['urls_in_description'] = df['description'].apply(urls_in_str)
    feature_df['def_image'] = (df['default_profile_image'] == 'True').astype(int)
    feature_df['def_profile'] = (df['default_profile'] == 'True').astype(int)

    #KANTEPE 
    feature_df['is_protected'] = (df['protected'] == 'True').astype(int)
    feature_df['screen_name_entropy'] = df['screen_name'].apply(get_str_entropy)
    feature_df['tweet_freq'] = df.apply(lambda x: frequency(x['created_at'], x['statuses_count']), axis=1)

    #KNAUTH
    feature_df['is_geo_enabled'] = (df['geo_enabled'] == 'True').astype(int)
    feature_df["favourites_count"] = df['favourites_count'].astype(int)
    feature_df['use_background_img'] = (df['background_image'] == 'True').astype(int)
    feature_df['lev_dist_name_screeName'] = df.apply(lambda x: lev_name_screenName(x['name'], x['screen_name']), axis=1)

    #SGBOT 
    feature_df['followers_growth_rate'] = df.apply(lambda x: frequency(x['created_at'], x['followers_count']), axis=1)
    feature_df['followings_growth_rate'] = df.apply(lambda x: frequency(x['created_at'], x['friends_count']), axis=1)
    feature_df['favourites_growth_rate'] = df.apply(lambda x: frequency(x['created_at'], x['favourites_count']), axis=1)

    feature_df = feature_df.drop(to_ignore,axis=1).reset_index(drop=True)  

    if normalize : 
        columns_names = feature_df.columns.difference(['account_id','label','split'])
        #feature_df[columns_names] = feature_df[columns_names].apply(zscore)
        train_df = feature_df[feature_df['split'] =='train']
        scaler = StandardScaler()
        scaler.fit(train_df[columns_names])

        feature_df[columns_names] = scaler.transform(feature_df[columns_names])


    return feature_df 



def process_dataset(dataset_v : str, kwargs = None) -> pd.DataFrame:

    dataset_path_v1 = glob.DATA_FOLDER / 'processed_dataset_v1.pkl'
    dataset_path_v2 = glob.DATA_FOLDER / 'processed_dataset_v2.pkl'
    dataset_path_v3 = glob.DATA_FOLDER / 'processed_dataset_v3.pkl'
    dataset_path_v4 = glob.DATA_FOLDER / 'processed_dataset_v4.pkl'
    dataset_path_v5 = glob.DATA_FOLDER / 'processed_dataset_v5.pkl'

    match dataset_v : 
        case 'v1': 
            print('starting dataset processing')
            if not os.path.exists(dataset_path_v1) or glob.force_processing:
                tweets_df, account_df = loadData()
                dataset_df = process_dataset_v1(tweets_df,dataset_path_v1)
            else : 
                print('found already processed dataset in data folder, retrieving the file...')
                dataset_df = pd.read_pickle(dataset_path_v1)
                print('dataset loaded in Dataframe')
            
            return dataset_df
        
        case 'v2':
            print('starting dataset processing')
            if not os.path.exists(dataset_path_v2) or glob.force_processing:
                if not os.path.exists(dataset_path_v1) or glob.force_processing:
                    tweets_df, account_df = loadData()
                    dataset_df = process_dataset_v1(tweets_df,dataset_path_v1)
                else : 
                    dataset_df = pd.read_pickle(dataset_path_v1)
            
                dataset_df = process_dataset_v2(dataset_df,dataset_path_v2, **kwargs['v2'])
            else : 
                print('found already processed dataset in data folder, retrieving the file...')
                dataset_df = pd.read_pickle(dataset_path_v2)
                print('dataset loaded in Dataframe')

            return dataset_df
        
        case 'v3':
            print('starting dataset processing')
            if not os.path.exists(dataset_path_v3) or glob.force_processing:

                if not os.path.exists(dataset_path_v1) or glob.force_processing:
                    tweets_df, account_df = loadData()
                    dataset_df = process_dataset_v1(tweets_df,dataset_path_v1)
                else : 
                    dataset_df = pd.read_pickle(dataset_path_v1)
                
                dataset_df = process_dataset_v3(dataset_df,dataset_path_v3,**kwargs)
            else : 
                print('found already processed dataset in data folder, retrieving the file...')
                dataset_df = pd.read_pickle(dataset_path_v3)
                print('dataset loaded in Dataframe')

            return dataset_df

        case 'v4':
            print('starting dataset processing')
            if not os.path.exists(dataset_path_v4) or glob.force_processing:

                v3_dataset = process_dataset('v3',kwargs['v3'])
                tweets_df, account_df = loadData()
                account_df_processed = process_account_dataset(account_df,**kwargs['v4'])

                dataset_df = pd.merge(v3_dataset, account_df_processed, on=['account_id','label','split'])
                print('saving processed dataset to file')
                dataset_df.to_pickle(dataset_path_v4)   #save to file

                #assert len(dataset_df) == len(account_df_processed), 'error while merging dataframes'

            else : 
                print('found already processed dataset in data folder, retrieving the file...')
                dataset_df = pd.read_pickle(dataset_path_v4)
                print('dataset loaded in Dataframe')
            
            return dataset_df

        case 'v5':
            print('starting dataset processing')
            if not os.path.exists(dataset_path_v5) or glob.force_processing:
                v3_dataset = process_dataset('v3',kwargs['v3'])
                tweets_df, account_df = loadData()
                account_df_processed = process_account_dataset(account_df,**kwargs['v5'])

                dataset_df = pd.merge(v3_dataset, account_df_processed, on=['account_id','label','split'])
                print('saving processed dataset to file')
                dataset_df.to_pickle(dataset_path_v5)   #save to file

            else : 
                print('found already processed dataset in data folder, retrieving the file...')
                dataset_df = pd.read_pickle(dataset_path_v5)
                print('dataset loaded in Dataframe')
            
            return dataset_df
        
        case 'account':

            print('processing dataset')
            tweets_df, account_df = loadData()
            account_df_processed = process_account_dataset(account_df,False)

            return account_df_processed

        case _ : 
            return NotImplementedError()