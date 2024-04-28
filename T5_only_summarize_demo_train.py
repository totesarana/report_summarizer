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
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk

from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForSeq2Seq

checkpoint = 't5-base'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding='max_length')
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
rouge_score = evaluate.load("rouge")
nltk.download("punkt")

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = 'cpu'


def read_in_dataset(jsonfile='GT.json'):
    with open(jsonfile, 'r') as fid:
        mydataset = json.load(fid)

    trimmed_dataset = {'suid': [], 'text': [], 'summary': []}

    for suid in mydataset:
        if len(mydataset[suid].keys()) > 0:
            trimmed_dataset['suid'].append(suid)
            trimmed_dataset['text'].append(mydataset[suid]['original_report'])
            trimmed_dataset['summary'].append(mydataset[suid]['summary'])
    return trimmed_dataset


def tokenize_function(example):
    text = ['summarize: ' + t for t in example['text']]
    return tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")


def preprocess_function(examples):

    examples['text'] = [text.replace('\\n', '.') for text in examples['text']]
    examples['text'] = [text.replace('<br>', '.') for text in examples['text']]
    examples['text'] = ['summarize: '+text for text in examples['text']]
    examples['summary'] = [text.replace('\\n', '.') for text in examples['summary']]
    examples['summary'] = [text.replace('<br>', '.') for text in examples['summary']]

    model_inputs = tokenizer(
        examples["text"],
        max_length=768,
        truncation=True,
    )
    labels = tokenizer(
        examples["summary"], max_length=128, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


if __name__ == '__main__':
    mydataset = read_in_dataset()
    mydataset['summary'] = [k['0'] for k in mydataset['summary']]
    mydataset = hf_datasets.Dataset.from_dict(mydataset)
    train_test = mydataset.train_test_split(test_size=0.2, seed=66)
    train_val = train_test['train'].train_test_split(test_size=0.2, seed=88)
    tokenized_datasets = train_val.map(preprocess_function, batched=True)

    # set up trainer
    batch_size = 2
    num_train_epochs = 8

    logging_steps = 5#len(tokenized_datasets["train"]) // batch_size

    args = Seq2SeqTrainingArguments(
        output_dir=f"{checkpoint}-finetuned-20230907",
        evaluation_strategy="epoch",
        learning_rate=5.6e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_dir='./logs', #directory for storing logs
        logging_strategy='steps',
        logging_steps=logging_steps,
        push_to_hub=False,
        save_strategy='steps'
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    tokenized_datasets = tokenized_datasets.remove_columns(
        train_val["train"].column_names
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()