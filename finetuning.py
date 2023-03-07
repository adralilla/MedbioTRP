import random
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
)
import pandas as pd
from datasets import Dataset
import evaluate
import numpy as np

training_set_bact = "/scratch/fastscratch/NBL/training_datasets/inherit_phage/IDS_InheritDBTraining_LabelBac_k6_lcashift1_Ls512"
training_set_phage = "/scratch/fastscratch/NBL/training_datasets/inherit_phage/IDS_InheritDBTraining_LabelPha_k6_lcashift1_Ls512"

eval_set_bact = "/scratch/fastscratch/NBL/training_datasets/inherit_phage/IDS_InheritDBValidation_LabelBac_k6_lcashift1_Ls512"
eval_set_phage = "/scratch/fastscratch/NBL/training_datasets/inherit_phage/IDS_InheritDBValidation_LabelPha_k6_lcashift1_Ls512"

model_path = "./models/checkpoint-61500/"
n = 10000
k = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()


with open(training_set_bact) as f:
    tmp = f.readlines()
records_train_bact = list()
for x in tmp:
    records_train_bact.append([int(y) for y in x.split(" ")])

records_train_bact = pd.DataFrame(
    {"input_ids": random.sample(records_train_bact, n), "label": 0}
)

with open(training_set_phage) as f:
    tmp = f.readlines()
records_train_phage = list()
for x in tmp:
    records_train_phage.append([int(y) for y in x.split(" ")])

records_train_phage = pd.DataFrame(
    {"input_ids": random.sample(records_train_phage, n), "label": 1}
)

with open(eval_set_bact) as f:
    tmp = f.readlines()
records_eval_bact = list()
for x in tmp:
    records_eval_bact.append([int(y) for y in x.split(" ")])

records_eval_bact = pd.DataFrame(
    {"input_ids": random.sample(records_eval_bact, k), "label": 0}
)

with open(eval_set_phage) as f:
    tmp = f.readlines()
records_eval_phage = list()
for x in tmp:
    records_eval_phage.append([int(y) for y in x.split(" ")])

records_eval_phage = pd.DataFrame(
    {"input_ids": random.sample(records_eval_phage, k), "label": 1}
)
train_dataset = Dataset.from_pandas(
    pd.concat([records_train_bact, records_train_phage], ignore_index=True)
)

eval_dataset = Dataset.from_pandas(
    pd.concat([records_eval_bact, records_eval_phage], ignore_index=True)
)

metric = evaluate.load("accuracy")
config = AutoConfig.from_pretrained(f"{model_path}config.json")
model = AutoModelForSequenceClassification.from_config(config)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
tokenizer = AutoTokenizer.from_pretrained(model_path)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./models/")
