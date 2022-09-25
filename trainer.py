# %%
import os, sys
trainer_path = os.path.abspath(__file__)
salmon_dir = os.path.dirname(trainer_path)
base_dir = os.path.join(salmon_dir, 'output')
sys.path.append(salmon_dir)

from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainer)

from datasets import Dataset
from datasets import load_metric

from typing import List
from datetime import datetime

from preprocess import get_train_df, get_test_df, get_data_from_txt, preprocess
from data import split_train_valid, get_hf_ds
from data_augmentation import get_augmented_df

import random
import json
import numpy as np
import torch
import pandas as pd

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
        model_name = 'KETI-AIR/ke-t5-base'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.add_tokens(add_token)
    
    return tokenizer


def get_pretrain_model(model_path: str,
                       tokenizer_size: AutoTokenizer) -> AutoModelForSeq2SeqLM:
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    except Exception as e:
        print(e, 'model_path not found, trying to load model from huggingface hub')
        model_name = 'KETI-AIR/ke-t5-base'
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
    
    batch_size = 16
    gradient_accumulation = 2 # 2가 최적
    num_train_epochs = 3
    learning_rate = 5e-4
    # 워밍업 스텝 비율
    # 20000개 데이터중에, train valid fold 나누는 기준
    # 10일 시 10개 폴드이므로, train:18000, valid:2000 비율이 최적
    train_valid_n_split = 10
    # 매 n+1마다 원본 파일을 기준으로 2배 증강함
    # n이 3이면 데이터 3배 증가
    num_data_augment = 0
    # input text의 전체 길이 설정, 97은 input text의 맥스값으로 설정한 상태 -> prefix때문에 100으로 변경
    input_max_length = 100
    # label text의 전체 길이 설정, 88은 train label의 맥스값으로 설정한 상태 -> prefix때문에 90으로 변경
    label_max_length = 90
    beam_search = True
    
    # 파일들이 save 될 장소를 설정해야함, 안그러면 겹칠 수 있음!
    base_path = base_dir
    # 띄어쓰기 필요없음
    prefix = 'extract entity:'
    label_prefix = 'extracted entity:'
    
    log_message = f'input text: {prefix}: \n, label text: {label_prefix}'
    
    base_model_path = '/home/work/team03/model/kt-ulm-base'
    small_model_path = '/home/work/team03/model/kt-ulm-small'
    
    # 원하는 모델로 model_path에 입력
    model_path = base_model_path
    tokenizer_path = '/home/work/team03/model/kt-ulm-base'
    
    
    tokenizer = get_pretrain_tokenizer(model_path=tokenizer_path)
    model = get_pretrain_model(model_path=model_path, tokenizer_size=tokenizer)
    
    train_df = get_train_df(prefix=prefix, label_prefix=label_prefix)
    test_df = get_test_df(prefix=prefix, label_prefix=label_prefix)
    
    train_df, valid_df = split_train_valid(preprocessed_df=train_df, n_split=train_valid_n_split)
    #train_df = get_augmented_df(preprocessed_df = train_df, augment_number=num_data_augment)
    
    train_hf_ds = get_hf_ds(train_df)
    valid_hf_ds = get_hf_ds(valid_df)
    test_hf_ds = get_hf_ds(test_df)
    
    tokenized_train_ds = train_hf_ds.map(tokenize_fn, batched=True, remove_columns=train_hf_ds.column_names)
    tokenized_valid_ds = valid_hf_ds.map(tokenize_fn, batched=True, remove_columns=valid_hf_ds.column_names)
    tokenized_test_ds = test_hf_ds.map(tokenize_fn, batched=True, remove_columns=test_hf_ds.column_names)

# %%
    data_collator = DataCollatorForSeq2Seq(tokenizer, model)
    
    warm_up_steps = int(((train_df.shape[0] * num_train_epochs) // (batch_size * gradient_accumulation)) * 0.2)
    logging_steps = int(((train_df.shape[0] * num_train_epochs) // (batch_size * gradient_accumulation)) * 0.2)
    save_steps = int(((train_df.shape[0] * num_train_epochs) // (batch_size * gradient_accumulation)) * 0.2)
    eval_steps = int(((train_df.shape[0] * num_train_epochs) // (batch_size * gradient_accumulation)) * 0.2)
    
    print('logging_steps: ', logging_steps)
    print('save_steps: ', save_steps)
    print('eval_steps: ', eval_steps)
    timenow = datetime.now()
    time_format = timenow.strftime('%d-%H:%M')
    
    file_name = f'{prefix}_{label_prefix}' + 'grad_accmul' + str(gradient_accumulation) + '_' + 'n_split' + str(train_valid_n_split) + '_' + 'n_augment' + str(num_data_augment)  + '_' + str(batch_size) + '_' + str(num_train_epochs) + '_' + model_path.split('/')[-1] + '_' + str(learning_rate) + '_' 'date' + time_format
    
    file_path = os.path.join(base_path, file_name)
    
    debug = False
    if debug == True:
        file_name = 'debug' + '_' + file_name
        
    
    args = Seq2SeqTrainingArguments(
    output_dir=file_path,
    evaluation_strategy="steps",
    save_strategy='steps',
    save_steps=save_steps,
    eval_steps=eval_steps,
    learning_rate=learning_rate,
    gradient_accumulation_steps=gradient_accumulation,
    # gradient_checkpointing=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size * 8,
    weight_decay=0.01,
    warmup_steps=warm_up_steps,
    seed=seed,
    data_seed=seed,
    #bf16=True,
    half_precision_backend='auto',
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
    greater_is_better=True,
    generation_max_length=label_max_length,
    generation_num_beams=beam_search, # 효과 x 
    # sortish_sampler=False,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    #report_to='wandb', # Wandb 사용 시 api키 등록이 필요함
    #run_name=file_name, # 검증 과정을 위해 주석처리
    
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
    
    message_path = os.path.join(file_path, 'readme.md')
    with open(message_path, 'w') as f:
        f.write(log_message)
    
    decoded_preds, decoded_labels = decode_prediction(predictionoutput=prediction_output, tokenizer=tokenizer)
    
    result_df = pd.DataFrame({"pred":decoded_preds, "label":decoded_labels})
    submission_df = pd.DataFrame({'input_sentence':test_df.input_sentence, 'pred':decoded_preds})
    
    result_df_path = os.path.join(file_path, 'test_pred_label.csv')
    submission_df_path = os.path.join(file_path, 'submission.csv')
    
    result_df.to_csv(result_df_path, index=False)
    submission_df.to_csv(submission_df_path, index=False)
    


