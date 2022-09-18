from collections import defaultdict
from tqdm import tqdm
from typing import TypeVar, List, Dict
import random
import pandas as pd
from copy import deepcopy

DataFrame = TypeVar('DataFrame')

def get_entity_dict(preprocessed_df: DataFrame)-> Dict[str, List[str]]:
    
    entity_dict = defaultdict(list)

    for row_object in tqdm(preprocessed_df.itertuples()):
        for word in row_object.train_label:
            # aa:PS shape
            temp_list = word.split(':')
            entity_dict[temp_list[-1]].append(temp_list[0])
    
    # drop duplicated words
    for i in entity_dict.keys():
        entity_dict[i] = list(set(entity_dict[i]))
    
    # entity dict result print
    for i in entity_dict.keys():
        print(i,' : ' ,len(entity_dict[i]))
    
    return entity_dict

def augment_entity_base(preprocessed_df: DataFrame) -> DataFrame:
    
    entity_dict = get_entity_dict(preprocessed_df=preprocessed_df)
    
    changed_entity_list = []
    changed_sentence_list = []
    
    for row_object in tqdm(preprocessed_df.itertuples()):
        row_list = []
        temp_sentence = deepcopy(row_object.org_sentence)
        for word in row_object.train_label:
            
            # [name, entity] split
            temp_list = word.split(':')
            temp_name = deepcopy(temp_list[0])
            
            # access to entity dict, choose one random word
            random_word = random.choice(entity_dict[temp_list[-1]])
            # change entity name 
            temp_sentence = temp_sentence.replace(temp_name, random_word, 1)
            temp_list[0] = random_word
            
            # create name:entity
            temp_entity = ':'.join(temp_list)
            # add
            row_list.append(temp_entity)

            
        changed_sentence_list.append(temp_sentence)
        changed_entity_list.append(row_list)
    
    augmented_df = pd.DataFrame({'augmented_sentence':changed_sentence_list, 'augmented_label':changed_entity_list})
    
    return augmented_df

