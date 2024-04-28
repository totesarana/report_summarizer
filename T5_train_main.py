import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import os
import numpy as np
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk

from nltk.tokenize import sent_tokenize
from transformers import DataCollatorForSeq2Seq
from Data import prepare
import argparse


checkpoint = 't5-base'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=1024, padding='max_length')
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
rouge_score = evaluate.load("rouge")
nltk.download("punkt")

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = 'cpu'


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


def preprocess_function(examples):
    examples['input_text'] = [text.replace('\\n', '.') for text in examples['input_text']]
    examples['input_text'] = [text.replace('<br>', '.') for text in examples['input_text']]
    examples['target_text'] = [text.replace('\\n', '.') for text in examples['target_text']]
    examples['target_text'] = [text.replace('<br>', '.') for text in examples['target_text']]

    model_inputs = tokenizer(
        examples["input_text"],
        max_length=768,
        truncation=True,
    )
    labels = tokenizer(
        examples["target_text"], max_length=128, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GTjson', action='store', default='./Data/GT.json', help='the ground truth json file')
    parser.add_argument('--save_dir', required=True, help='provide save directory')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training and validation')
    parser.add_argument('--lr', default=5.6e-5, type=float, help='learning rate')
    parser.add_argument('--epoches', default=10, type=int, help='total number of epoches')
    parser.add_argument('--seed', default=42, type=int, help='the seed, by default it is 42')
    parser.add_argument("--fdriveinput", help='F drive address on the cloud (input folder)')
    parser.add_argument("--fdriveoutput", help='F drive address on the cloud (output folder)')
    args = parser.parse_args()

    jsonfile = args.GTjson
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    bs = args.batch_size
    lr = args.lr
    num_train_epochs = args.epoches
    seed = args.seed

    if args.fdriveinput:
        fdriveinput = args.fdriveinput
        fdriveoutput = args.fdriveoutput
        jsonfile = jsonfile.replace("F:", fdriveinput).replace("\\", "/")
        save_dir = save_dir.replace("F:", fdriveoutput).replace("\\", "/")
        print("Save directory is: " + save_dir)

    trimmed_dataset = prepare.read_in_dataset(jsonfile=jsonfile)
    split_dict, trimmed_dataset = prepare.split_on_suid_levels(save_dir, trimmed_dataset, seed=seed, save_split=True)
    train_dict = prepare.build_dict(trimmed_dataset, group='train')
    hf_train_dataset = prepare.build_hfDataset(train_dict)
    val_dict = prepare.build_dict(trimmed_dataset, group='val')
    hf_val_dataset = prepare.build_hfDataset(val_dict)
    tokenized_train_datasets = hf_train_dataset.map(preprocess_function, batched=True)
    tokenized_val_datasets = hf_val_dataset.map(preprocess_function, batched=True)

    batch_size = bs
    logging_steps = 5

    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(save_dir, f"{checkpoint}-finetuned"),
        evaluation_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_dir=os.path.join(save_dir, './logs'),  # directory for storing logs
        logging_strategy='steps',
        logging_steps=logging_steps,
        push_to_hub=False,
        save_strategy='steps'
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    tokenized_train_datasets = tokenized_train_datasets.remove_columns(
        hf_train_dataset.column_names
    )

    tokenized_val_datasets = tokenized_val_datasets.remove_columns(
        hf_val_dataset.column_names
    )

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_val_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == '__main__':
    main()