from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments
from typing import List
from datasets import Dataset
from preprocess import get_train_df
from data import get_train_valid_ds
import os, random
import numpy as np
import torch

# seed 고정
seed = 1514
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def get_pretrain_tokenizer(model_path: str = None, 
                           add_token: List[str] = [':DT', ':LC', ':OG', ':PS', ':QT', ':TI']):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)    
    except Exception as e:
        print(e, 'model_path not found, trying to load model from huggingface hub')
        model_name = 'klue/ke-t5-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.add_tokens(add_token)
    
    return tokenizer

def get_pretrain_model(model_path: str,
                       tokenizer_size: AutoTokenizer):
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    except Exception as e:
        print(e, 'model_path not found, trying to load model from huggingface hub')
        model_name = 'klue/ke-t5-base'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    model.resize_token_embeddings(len(tokenizer_size))
    
    return model

def tokenize_fn(ds, input_max_length: int = 50, label_max_length: int = 8):
    model_inputs = tokenizer(ds['input_sentence'], max_length=input_max_length, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            ds['train_label'], max_length=label_max_length, truncation=True,
        )['input_ids']
    
    model_inputs['labels'] = labels[0]
    return model_inputs

def get_token_ds(hf_ds: Dataset, input_max_length: int = 50, label_max_length: int = 8):
    tokenized_hf_ds = hf_ds.map(lambda x: tokenize_fn(x, input_max_length, label_max_length), remove_columns=[hf_ds.columns])
    return tokenized_hf_ds
    
if __name__ == '__main__':
    train_df = get_train_df()
    train_hf_ds, valid_hf_ds = get_train_valid_ds(train_df, n_split=4)
    model_path = '/root/klue_ner/pretrained_model/ke-t5-base'
    tokenizer = get_pretrain_tokenizer(model_path=model_path)
    model = get_pretrain_model(model_path=model_path, tokenizer_size=len(tokenizer))
    tokenized_train_ds = get_token_ds(train_hf_ds, input_max_length=50, label_max_length=8)
    tokenized_valid_ds = get_token_ds(valid_hf_ds, input_max_length=50, label_max_length=8)
    
    batch_size = 32
    num_train_epochs = 20
    logging_steps = len(tokenized_train_ds) // batch_size
    
    args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-{data_path.split('/')[-1]}",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
)