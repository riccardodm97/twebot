# IMPORTS 
from pathlib import Path 
import os 
import json 

import pandas as pd
import numpy as np
from collections import OrderedDict, namedtuple

import wandb 
from datetime import datetime
import pytz
import src.globals as glob

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