import torch
import numpy as np
from datasets import load_metric
import re

bleu_metric = load_metric('sacrebleu')

def decode_prediction(predictionoutput, tokenizer):
    preds, labels = predictionoutput.predictions, predictionoutput.label_ids
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    return decoded_preds, decoded_labels

def get_f1_acc(preds, labels):
    p = re.compile(":[A-Za-z]+")
    result = p.finditer(preds)
    s=0
    e=-1
    pred_list = list()
    pred_list.append([list(), ''])
    for i, r in enumerate(result):
        e=r.span()[0]
        pred_list[i][0].extend(preds[s:e].split(' '))
        pred_list[i][1] += preds[r.span()[0]:r.span()[1]]
        s=r.span()[1]+1
        pred_list.append([list(), ''])    
    del pred_list[-1] # pred list 생성
    
    p = re.compile(":[A-Za-z]+")
    result = p.finditer(labels)
    s=0
    e=-1
    label_list = list()
    label_list.append([list(), ''])
    for i, r in enumerate(result):
        e=r.span()[0]
        label_list[i][0].extend(labels[s:e].split(' '))
        label_list[i][1] += labels[r.span()[0]:r.span()[1]]
        s=r.span()[1]+1
        label_list.append([list(), ''])    
    del label_list[-1] # label list 생성
    
    visit = np.zeros((len(label_list), ))
    true_positive = 0
    for pred in pred_list:
        pred_substring_list = pred[0]
        pred_tag = pred[1]
        flag = False
        for i, label in enumerate(label_list):
            if visit[i] > 0:
              continue
            label_substring_list = label[0]
            label_tag = label[1]
            j = sum([pred in label_substring_list for pred in pred_substring_list])
            if j > 0 and j >= len(pred_substring_list)-2:
                if label_tag == pred_tag:
                   true_positive += 1
                   visit[i] += 1
                break # true_positive 구하기
    
    flase_positive = len(pred_list) - true_positive

    if true_positive + flase_positive == 0 or len(label_list) == 0:
        return 0, 0
    
    precision = true_positive / len(pred_list)

    true_positive_label = 0
    for label in label_list:
        label_substring_list = label[0]
        label_tag = label[1]
        flag = False
        for pred in pred_list:
            pred_substring_list = pred[0]
            pred_tag = pred[1]
            j = sum([label in pred_substring_list for label in label_substring_list])
            if j > 0 and j >= len(label_substring_list)-2:
                if label_tag == pred_tag:
                   true_positive_label += 1
                break # true_positive 구하기

    false_negative = len(label_list) - true_positive_label
    recall = true_positive_label / len(label_list)
    
    if precision == 0 or recall == 0 :
        return 0, 0
    f1 = 2 / (1/precision + 1/recall)

    # print(pred_list)
    # print(label_list)
    # print('len(label_list)', len(label_list))
    # print('len(pred_list)', len(pred_list))
    # print('true_positive', true_positive)
    # print('flase_positive', flase_positive)
    # print('false_negative', false_negative)
    # print('precision', precision)
    # print('recall', recall)
    # print('f1', f1)
    
    if len(label_list) == 0:
        return 0, 0
    acc = true_positive_label / len(label_list)
    return f1, acc

def compute_f1_acc(decoded_preds, decoded_labels):
    
    f1_list = []
    acc_list = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        f1, acc = get_f1_acc(preds=pred, labels=label)
        f1_list.append(f1)
        acc_list.append(acc)
    
    average_f1 = sum(f1_list) / len(f1_list)
    average_acc = sum(acc_list) / len(acc_list)
    
    return {'f1':average_f1, 'acc':average_acc}
    

def compute_bleu(decoded_preds, decoded_labels):

    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {"bleu": result["score"]}

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