import random, os
import numpy as np
import torch

from sklearn.feature_extraction.text import CountVectorizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from typing import TypeVar, Tuple
from datasets import Dataset

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

def split_train_valid(preprocessed_df: DataFrame, n_split: int = 4) -> Tuple[DataFrame, DataFrame]:
    
    # stratified sampling by entity distribution
    CV = CountVectorizer(binary=True)
    # [0,1,0,1,0,1] shape embedding
    entity_embeddings = CV.fit_transform(preprocessed_df.joined_entity).toarray()
    # to set 20% of the whole data(about 5000), default n_split is 4
    MSKF = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=1514)
    train_idx, valid_idx = next(iter(MSKF.split(preprocessed_df.input_sentence, entity_embeddings)))
    train_df = preprocessed_df.iloc[train_idx].reset_index(drop=True)
    valid_df = preprocessed_df.iloc[valid_idx].reset_index(drop=True)
    print('train_data: ', train_idx.shape)
    print('valid_data: ', valid_idx.shape)
    return train_df, valid_df

def get_hf_ds(df: DataFrame) -> Dataset:
    # dataframe -> huggingface dataset
    hf_ds = Dataset.from_pandas(df[['input_sentence','train_label']])
    
    return hf_ds

def get_train_valid_ds(preprocessed_df : DataFrame, n_split: int = 4) -> Tuple[Dataset, Dataset]:
    # do not use this function for test data
    train_df, valid_df = split_train_valid(preprocessed_df,n_split=n_split)
    train_hf_ds = get_hf_ds(train_df)
    valid_hf_ds = get_hf_ds(valid_df)
    
    return train_hf_ds, valid_hf_ds


if __name__ == '__main__':
    print('a')