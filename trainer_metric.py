import torch
import numpy as np
from datasets import load_metric

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
    
    tags = ['QT', 'DT', 'PS', 'LC', 'TI', 'OG']
    pred_list = list()
    pred_list.append([list(), ''])
    c = 0
    for pred in preds.split(':'):
        if pred not in tags:
            pred_list[c][0].append(pred)
        else:
            pred_list[c][1] += pred
            pred_list.append([list(), ''])
            c += 1
    del pred_list[-1] # pred list 생성
    label_list = list()
    label_list.append([list(), ''])
    c = 0
    for label in labels.split(':'):
        if label not in tags:
            label_list[c][0].append(label)
        else:
            label_list[c][1] += label
            label_list.append([list(), ''])
            c += 1
    del label_list[-1] # label list 생성
    
    # print(pred_list)
    # print(label_list)
    
    true_positive = 0
    for label in label_list:
        label_substring_list = label[0]
        label_tag = label[1]
        flag = False
        for pred in pred_list:
            pred_substring_list = pred[0]
            pred_tag = pred[1]
            if sum([label in pred_substring_list for label in label_substring_list]) > 0:
                if label_tag == pred_tag:
                   true_positive += 1
                break # true_positive 구하기
    
    flase_positive = len(pred_list) - true_positive
    false_negative = len(label_list) - true_positive
    
    if true_positive + flase_positive == 0 or len(label_list) == 0:
        return 0, 0
    
    precision = true_positive / (true_positive + flase_positive)
    recall = true_positive / len(label_list)
    
    if precision == 0 or recall == 0 :
        return 0, 0
    f1 = 2 / (1/precision + 1/recall)
    
#     print('len(label_list)', len(label_list))
#     print('len(pred_list)', len(pred_list))
#     print('true_positive', true_positive)
#     print('flase_positive', flase_positive)
#     print('false_negative', false_negative)
#     print('precision', precision)
#     print('recall', recall)
#     print('f1', f1)
    
    if len(label_list) == 0:
        return 0, 0
    acc = true_positive / len(label_list)
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