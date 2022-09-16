def compute_metric(preds, labels):
    tags = [':QT', ':DT', ':PS', ':LC', ':TI', ':OG']
    pred_list = list()
    pred_list.append([list(), ''])
    c = 0
    for pred in preds.split(' '):
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
    for label in labels.split(' '):
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
        return None
    
    precision = true_positive / (true_positive + flase_positive)
    recall = true_positive / len(label_list)
    
    if precision == 0 or recall == 0 :
        return None
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
        return None
    return f1, true_positive / len(label_list)
