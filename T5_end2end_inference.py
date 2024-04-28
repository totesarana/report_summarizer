import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import json
import random
import numpy as np
import datasets as hf_datasets
from torch.utils.data import DataLoader
from transformers import pipeline
from tqdm import tqdm
import evaluate


checkpoint = 't5-base'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding='max_length')
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = 'cpu'

def read_in_dataset(jsonfile = 'GT.json'):
    with open(jsonfile, 'r') as fid:
        mydataset = json.load(fid)

    trimmed_dataset = {'suid':[], 'text':[], 'summary':[]}

    for suid in mydataset:
        if len(mydataset[suid].keys())>0:
            trimmed_dataset['suid'].append(suid)
            trimmed_dataset['text'].append(mydataset[suid]['original_report'])
            trimmed_dataset['summary'].append(mydataset[suid]['summary'])
    return trimmed_dataset


def split_train_val_test(alldataset, seed=114514, train_val_test_ratio = [0.7, 0.2, 0.1]):
    N = len(alldataset['suid'])
    pool_nums = list(range(N))
    train_ratio = train_val_test_ratio[0]
    val_ratio = train_val_test_ratio[1]
    test_ratio = train_val_test_ratio[2]
    # Calculate the number of elements for each group based on the ratios
    num_train = int(N * train_ratio)
    num_val = int(N * val_ratio)
    num_test = N - num_train - num_val
    # Shuffle the input list
    random.seed(seed)
    random.shuffle(pool_nums)
    # Split the list into three groups using slicing
    train_IDs = pool_nums[:num_train]
    val_IDs = pool_nums[num_train : (num_train+num_val)]
    test_IDs = pool_nums[(num_train+num_val) :]
    #Construct train, val and test dataset
    train_data = {'suid': [], 'text': [], 'summary': []}
    val_data = {'suid': [], 'text': [], 'summary': []}
    test_data = {'suid': [], 'text': [], 'summary': []}
    for idx in train_IDs:
        train_data['suid'].append(alldataset['suid'][idx])
        train_data['text'].append(alldataset['text'][idx])
        train_data['summary'].append(alldataset['summary'][idx])
    for idx in val_IDs:
        val_data['suid'].append(alldataset['suid'][idx])
        val_data['text'].append(alldataset['text'][idx])
        val_data['summary'].append(alldataset['summary'][idx])
    for idx in test_IDs:
        test_data['suid'].append(alldataset['suid'][idx])
        test_data['text'].append(alldataset['text'][idx])
        test_data['summary'].append(alldataset['summary'][idx])
    return train_data, val_data, test_data


def tokenize_function(example):
    text = ['summarize: ' + t for t in example['text']]
    return tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")


def inference(test_dataset, checkpoint = 't5-base'):
    hf_test_dataset = hf_datasets.Dataset.from_dict(test_dataset)

    #tokenized_dataset = hf_test_dataset.map(tokenize_function, batched=True)
    #create the pipeline
    summarizer = pipeline("summarization", tokenizer=checkpoint, framework='pt', model=checkpoint)

    batch_size = 8
    dataloader = DataLoader(hf_test_dataset, batch_size=batch_size)
    references = []
    generated_outputs = []
    input_texts = []
    rouge_metric = evaluate.load('rouge')
    res_obj = {'suid': [], 'orig_text': [], \
               'generated_summary': [], 'GPT4_summary': [],
               'rouge_score': []}

    for idx, batch_data in enumerate(tqdm(dataloader)):
        suids = batch_data['suid']
        texts = batch_data['text']
        texts = ['summarize: ' + t for t in texts]
        input_texts += texts
        refs = batch_data['summary']
        outputs = summarizer(texts)
        for i in range(len(texts)):
            res_obj['generated_summary'].append(outputs[i]['summary_text'])
            #new_refs = []
            #for j in range(4):
            #    new_refs.append(refs[str(j)][i])
            #res_obj['GPT4_summary'].append(new_refs)
            res_obj['GPT4_summary'].append(refs['0'][i])
            res_obj['orig_text'].append(texts[i][11:])
            res_obj['suid'].append(suids[i])
            print(outputs[i]['summary_text'])

            rouge_scores = rouge_metric.compute(predictions=[outputs[i]['summary_text']], \
                                        references=[refs['0'][i]])
            res_obj['rouge_score'].append(rouge_scores)
        with open('baseline_t5.obj', 'w') as fid:
            json.dump(res_obj, fid)



if __name__ == '__main__':
    mydataset = read_in_dataset()
    train_dataset, val_dataset, test_dataset = split_train_val_test(alldataset=mydataset)
    inference(test_dataset, checkpoint=checkpoint)
