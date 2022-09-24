import pandas as pd
import numpy as np
import random, os

import torch

from typing import TypeVar, Tuple
import re
from copy import deepcopy

# seed 고정
seed = 1514
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

DataFrame = TypeVar('DataFrame')

file_path = os.path.abspath(__file__)
salmon_dir = os.path.dirname(file_path)
data_dir = os.path.join(salmon_dir, 'raw_file')
train_path = os.path.join(data_dir, 'klue_ner_train_80.txt')
test_path = os.path.join(data_dir, 'klue_ner_test_20.txt')

# extract NER tag from sentence for label
def find_all_label(sentence: str) -> Tuple[list, np.array, np.array]:
    # extract <aa:PS> shape string
    sentence = re.findall( r'<.*?\:.*?>', sentence)
    # remove '<', '>' 
    sentence = [i.replace('<','') for i in sentence]
    sentence = [i.replace('>','') for i in sentence]
    # name, entity split by ':'
    before_split = deepcopy(sentence)
    sentence = [i.split(':') for i in sentence]
    sentence_array = np.array(sentence)
    # create only name array
    ner_array = sentence_array[...,0]
    # create only entity array
    label_array = sentence_array[...,1]
    return before_split, ner_array, label_array
    
# remove NER tag from sentence for input    
def remove_label(sentence: str) -> str:
    # remove <aa:PS> shape string
    sentence = re.sub(r':.*?>','', sentence)
    sentence = re.sub(r'<','',sentence)
    # remove all except for korean, english, number, etc..
    sentence = re.sub(r'[^가-힣a-zA-Z0-9.%~]',' ' ,sentence)
    # strip space
    sentence = re.sub('[ ]+',' ',sentence)
    sentence = sentence.strip()
    return sentence

def get_data_from_txt(path: str) -> DataFrame:
    # Get original train data from txt file 
    with open(path, 'r', encoding='UTF8') as f:
        # because of useless spacing, last line remove
        sentence_list = f.read().split('\n')[:-1]
    df = pd.DataFrame({'sentence':sentence_list})
    return df

def preprocess(df: DataFrame, train: bool=True, prefix:str=''):
    
    if train == True:
        df.sentence[10212] = df.sentence[10212].replace('일녀<QT>', '<일녀:QT>')
    df['labels'] = df.sentence.apply(find_all_label)
    df['train_label'] = df.labels.apply(lambda x: x[0])
    df['train_label'] = df.train_label.apply(lambda x: ' '.join(x))
    df['name'] = df.labels.apply(lambda x: x[1])
    df['entity'] = df.labels.apply(lambda x: x[2])
    df['input_sentence'] = df.sentence.apply(remove_label)
    df['input_sentence'] = prefix + ' ' + df.input_sentence
    df['input_sentence'] = df.input_sentence.str.strip()
    df['joined_entity'] = df.entity.apply(lambda x: ' '.join(x))
    return df

def get_train_df(path: str = train_path, prefix:str=''):
    df = get_data_from_txt(path)
    preprocessed_df = preprocess(df=df, train=True, prefix=prefix)
    return preprocessed_df

def get_test_df(path: str = test_path, prefix:str=''):
    df = get_data_from_txt(path)
    preprocessed_df = preprocess(df=df, train=False, prefix=prefix)
    return preprocessed_df
    
if __name__ == '__main__':
    train_df = get_train_df()
    test_df = get_test_df()
    print(train_df.shape)
    print(test_df.shape)