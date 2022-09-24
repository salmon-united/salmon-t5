# %%
import sys
sys.path.append('/home/work/team03/salmon-t5')

from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainer)
import evaluate
from typing import List
from datasets import Dataset
from preprocess import get_train_df, get_test_df
from data import get_train_valid_ds, split_train_valid, get_hf_ds
from data_augmentation import get_augmented_df
import os, random
import pandas as pd
import pickle, json
import numpy as np
import torch
from datetime import datetime
from datasets import load_metric
from trainer_metric import compute_bleu, compute_f1_acc, decode_prediction


# seed 고정
seed = 1514
random.seed(seed)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def get_pretrain_tokenizer(model_path: str = None, 
                           add_token: List[str] = [':DT', ':LC', ':OG', ':PS', ':QT', ':TI']) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)    
    except Exception as e:
        print(e, 'model_path not found, trying to load model from huggingface hub')
        model_name = 'klue/ke-t5-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.add_tokens(add_token)
    
    return tokenizer


def get_pretrain_model(model_path: str,
                       tokenizer_size: AutoTokenizer) -> AutoModelForSeq2SeqLM:
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    except Exception as e:
        print(e, 'model_path not found, trying to load model from huggingface hub')
        model_name = 'klue/ke-t5-base'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    model.resize_token_embeddings(len(tokenizer_size))
    
    return model


def tokenize_fn(ds) -> dict:
    model_inputs = tokenizer(ds['input_sentence'], max_length=input_max_length, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            ds['train_label'], max_length=label_max_length, truncation=True,
        )['input_ids']
    
    model_inputs['labels'] = labels
    return model_inputs

def compute_metrics(eval_preds):
        
    preds, labels = eval_preds.predictions, eval_preds.label_ids
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    bleu_decoded_labels = [[label.strip()] for label in decoded_labels]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    metric_dict = {}
    bleu_dict = compute_bleu(decoded_preds=decoded_preds, decoded_labels=bleu_decoded_labels)
    f1_acc_dict = compute_f1_acc(decoded_preds=decoded_preds, decoded_labels=decoded_labels)
    
    metric_dict.update(bleu_dict)
    metric_dict.update(f1_acc_dict)
    
    return metric_dict

    
if __name__ == '__main__':
    
    batch_size = 64
    num_train_epochs = 5
    learning_rate = 5e-4
    # 워밍업 스텝 비율
    warm_up_ratio = 0.1
    # 20000개 데이터중에, train valid fold 나누는 기준
    # 10일 시 10개 폴드이므로, train:18000, valid:2000 비율
    train_valid_n_split = 10
    # 매 n+1마다 원본 파일을 기준으로 2배 증강함
    # n이 3이면 데이터 3배 증가
    num_data_augment = 0
    # input text의 전체 길이 설정, 97은 input text의 맥스값으로 설정한 상태
    input_max_length = 97
    # label text의 전체 길이 설정, 88은 train label의 맥스값으로 설정한 상태
    label_max_length = 88
    
    base_model_path = '/home/work/team03/model/kt-ulm-base'
    small_model_path = '/home/work/team03/model/kt-ulm-small'
    # 원하는 모델로 model_path에 입력
    model_path = base_model_path
    
    tokenizer = get_pretrain_tokenizer(model_path=model_path)
    model = get_pretrain_model(model_path=model_path, tokenizer_size=tokenizer)
    
    train_df = get_train_df()
    test_df = get_test_df()
    
    train_df, valid_df = split_train_valid(preprocessed_df=train_df, n_split=train_valid_n_split)
    train_df = get_augmented_df(preprocessed_df = train_df, augment_number=num_data_augment)
    
    train_hf_ds = get_hf_ds(train_df)
    valid_hf_ds = get_hf_ds(valid_df)
    test_hf_ds = get_hf_ds(test_df)
    
    tokenized_train_ds = train_hf_ds.map(tokenize_fn, batched=True, remove_columns=train_hf_ds.column_names)
    tokenized_valid_ds = valid_hf_ds.map(tokenize_fn, batched=True, remove_columns=valid_hf_ds.column_names)
    tokenized_test_ds = test_hf_ds.map(tokenize_fn, batched=True, remove_columns=test_hf_ds.column_names)

# %%
    data_collator = DataCollatorForSeq2Seq(tokenizer, model)
    
    logging_steps = len(tokenized_train_ds) // batch_size
    
    file_name = str(batch_size) + '_' + str(num_train_epochs) + '_' + model_path.split('/')[-1] + '_' + str(learning_rate)
    
    file_path = os.path.join(f"/home/work/team03/salmon-gu/log", file_name)
    
    debug = True
    if debug == True:
        file_name = 'debug' + '_' + file_name
        
    
    args = Seq2SeqTrainingArguments(
    output_dir=file_path,
    evaluation_strategy="epoch",
    save_strategy='epoch',
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    warmup_ratio=warm_up_ratio,
    seed=seed,
    data_seed=seed,
    load_best_model_at_end=True,
    generation_max_length=88,
    generation_num_beams=None, # 효과 x 
    # sortish_sampler=False,
    save_total_limit=2,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    report_to='wandb',
    run_name='wandb_test',
    
)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

# %%
    prediction_output = trainer.predict(tokenized_test_ds)
    
    print(prediction_output.metrics)
    metrics_path = os.path.join(file_path, 'bleu_f1_acc.json')
    with open(metrics_path, 'w') as f:
        json.dump(prediction_output.metrics, f, ensure_ascii=False, indent=4)
    
    decoded_preds, decoded_labels = decode_prediction(predictionoutput=prediction_output, tokenizer=tokenizer)
    
    result_df = pd.DataFrame({"pred":decoded_preds, "label":decoded_labels})
    submission_df = pd.DataFrame({'input_sentence':test_df.input_sentence, 'pred':decoded_preds})
    
    result_df_path = os.path.join(file_path, 'test_pred_label.csv')
    submission_df_path = os.path.join(file_path, 'submission.csv')
    
    result_df.to_csv(result_df_path, index=False)
    submission_df.to_csv(submission_df_path, index=False)
    


