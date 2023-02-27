import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import numpy as np
import evaluate

data_files = {
    "train": "train_dataset.csv",
    # "test": "test_dataset.csv",
    # "validation": "validation_dataset.csv",
}
raw_dataset = load_dataset("csv", data_files=data_files)
tokenizer = AutoTokenizer.from_pretrained("./models/checkpoint-61500/")


def tokenize_function(example):
    return tokenizer(
        example["sequence"],
        padding=True,
        return_tensors="pt",
        is_split_into_words=True,
        truncation=True,
        model_max_length=512,
    )


tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)


# def compute_metrics(eval_preds):
#     metric = evaluate.load(" ")
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


config = AutoConfig.from_json_file("./models/checkpoint-61500/config.json")
model = AutoModelForSequenceClassification.from_config(config)
# training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    # training_args,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics,
)
