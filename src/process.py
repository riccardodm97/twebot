
import re

import emoji
from nltk.tokenize import TweetTokenizer
from ttp import ttp 
from emot.core import emot 
import pandas as pd 
from string import punctuation
import nltk 
from nltk.corpus import stopwords
from scipy.stats import zscore 

import src.globals as glob 


def process_dataset_v1(dataframe : pd.DataFrame , save_path ) :

    parser = ttp.Parser(include_spans=True)
    emot_obj = emot()
    tk = TweetTokenizer(reduce_len=True,preserve_case=False)

    CASHTAG = "(?<!\S)\$[A-Z]+(?:\.[A-Z]+)?(?!\S)"   # to check  (?:\.[A-Z]+)?
    EMAIL = r"""[\w.+-]+@[\w-]+\.(?:[\w-]\.?)+[\w-]"""
    MONEY = "[$£][0-9]+(?:[.,]\d+)?[k+B]?|[0-9]+(?:[.,]\d+)?[k+B]?[$£]"  
    NUMBERS = r"""(?<!\S)(?:[+\-]?\d+(?:%|(?:[,/.:-]\d+[+\-]?)?))"""   # r"""(?:[+\-]?\d+[,/.:-]\d+[+\-]?)"""   
    HASHTAG = r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    HANDLES = r"""(?:@[\w_]+)""" 

    TO_REPLACE = [CASHTAG, EMAIL, MONEY, NUMBERS, HASHTAG, HANDLES]
    REPLACE_WITH = [' stock ',' email ',' money ',' number ',' hashtag ',' username ']


    def replace(word : str):
        if not word.isascii():
            return ['']
        elif bool(re.search(r'http[s]?|.com',word)):
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

    print('starting data processing')

    dataframe['processed_tweet'] = dataframe['tweet'].replace(TO_REPLACE,REPLACE_WITH,regex=True,inplace=False)
    dataframe['processed_tweet'] = dataframe['processed_tweet'].apply(further_process)

    dataframe = dataframe[dataframe['processed_tweet'].map(lambda x: len(x)) > 2].reset_index(drop=True)   

    dataframe['label'] = dataframe['label'].astype(float)  

    print('saving processed dataset to file')
    dataframe.to_pickle(save_path)   #save to file 
    
    return dataframe
        


def process_dataset_v2(dataframe : pd.DataFrame, save_path) :

        nltk.download('stopwords')
    
        sw = stopwords.words('english')

        df = dataframe.copy(deep=True)  

        def is_retweet(sentence_list : list):
            return float(sentence_list[0] == 'rt')

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

        column_names = ['url_c','tag_c','hashtag_c','cashtag_c','money_c','email_c','number_c','emoji_c','emoticon_c','len_tweet','stopwords_c','punct_c']

        df[column_names] = df[column_names].apply(zscore)

        print('saving processed dataset to file')
        df.to_pickle(save_path)   #save to file

        return df 

