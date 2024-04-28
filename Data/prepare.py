import re
from collections import Counter
import random
import json
import os
import datetime
import pandas as pd
import datasets as hf_datasets


def read_in_dataset(jsonfile='GT.json'):
    with open(jsonfile, 'r') as fid:
        mydataset = json.load(fid)
    trimmed_dataset = {'suid': [], 'text': [], 'summary': [], 'malignancy': []}
    pattern = r'^[A-Za-z]+$'
    for suid in mydataset:
        if len(mydataset[suid].keys()) > 0:
            trimmed_dataset['suid'].append(suid)
            trimmed_dataset['text'].append(mydataset[suid]['original_report'])
            trimmed_dataset['summary'].append(mydataset[suid]['summary']['0'])
            malignancy = list(mydataset[suid]['malignancy'].values())
            #malignancy = [re.sub(pattern, '', k) for k in malignancy]
            malignancy = [k.replace('\\n', '') for k in malignancy]
            malignancy = [k.replace(' ', '') for k in malignancy]
            counter = Counter(malignancy)
            most_common_answer = counter.most_common(1)
            trimmed_dataset['malignancy'].append(most_common_answer[0][0])
    return trimmed_dataset


def split_on_suid_levels(save_dir, trimmed_dataset, seed=42, save_split=True):
    suids = trimmed_dataset['suid']
    random.seed(seed)
    random.shuffle(suids)
    N1 = int(len(suids)*0.7)
    N2 = int(len(suids)*0.9)
    train_suids = suids[:N1]
    val_suids = suids[N1:N2]
    test_suids = suids[N2:]
    split_dict = {'train': train_suids, 'val': val_suids, 'test': test_suids}
    os.makedirs(os.path.join(save_dir, 'train_val_test_split_record'), exist_ok=True)
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"split_{formatted_datetime}.json"
    if save_split:
        with open(os.path.join(save_dir, 'train_val_test_split_record', file_name), 'w') as fid:
            json.dump(split_dict, fid)

    trimmed_dataset['group'] = [None]*len(trimmed_dataset['suid'])
    for i, suid in enumerate(suids):
        if suid in train_suids:
            trimmed_dataset['group'][i] = 'train'
        elif suid in val_suids:
            trimmed_dataset['group'][i] = 'val'
        elif suid in test_suids:
            trimmed_dataset['group'][i] = 'test'
        else:
            raise ValueError('train val test split is wrong')
    return split_dict, trimmed_dataset


def build_dict(trimmed_dataset, group):
    my_dict = {'input_text': [], 'target_text': []}
    assert(group in ['train', 'val', 'test'])
    for i in range(len(trimmed_dataset['suid'])):
        if trimmed_dataset['group'][i] != group:
            continue
        context = trimmed_dataset['text'][i].replace('\\n', '.')
        summary = trimmed_dataset['summary'][i].replace('\\n', '.')
        question = 'question: is there cancer or malignant nodule? '
        prefix = 'summarize: '
        my_dict['input_text'].append(prefix+context)
        my_dict['input_text'].append(question + 'context: ' + context)
        my_dict['target_text'].append(summary)
        my_dict['target_text'].append(trimmed_dataset['malignancy'][i])
    return my_dict


def build_hfDataset(my_dict):
    df = pd.DataFrame(my_dict)
    created_dataset = hf_datasets.Dataset.from_pandas(df)
    return created_dataset


if __name__ == '__main__':
    jsonfile = 'GT.json'
    trimmed_dataset = read_in_dataset(jsonfile=jsonfile)
    split_dict, trimmed_dataset = split_on_suid_levels(trimmed_dataset, seed=42, save_split=True)
    train_dict = build_dict(trimmed_dataset, group='train')
    hf_train_dataset = build_hfDataset(train_dict)
    print(type(hf_train_dataset))



