from evaluate import evaluator
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import pandas as pd
import random
from datasets import Dataset
import torch

eval_set_bact = "/scratch/fastscratch/NBL/training_datasets/inherit_phage/IDS_InheritDBValidation_LabelBac_k6_lcashift1_Ls512"
eval_set_phage = "/scratch/fastscratch/NBL/training_datasets/inherit_phage/IDS_InheritDBValidation_LabelPha_k6_lcashift1_Ls512"
model_path = "./models/model_march_5/"

k = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

with open(eval_set_bact) as f:
    tmp = f.readlines()
records_eval_bact = list()
for x in tmp:
    records_eval_bact.append([int(y) for y in x.split(" ")])

records_eval_bact = pd.DataFrame(
    {"input_column": random.sample(records_eval_bact, k), "label_column": 0}
)

with open(eval_set_phage) as f:
    tmp = f.readlines()
records_eval_phage = list()
for x in tmp:
    records_eval_phage.append([int(y) for y in x.split(" ")])

records_eval_phage = pd.DataFrame(
    {"input_column": random.sample(records_eval_phage, k), "label_column": 1}
)

eval_dataset = Dataset.from_pandas(
    pd.concat([records_eval_bact, records_eval_phage], ignore_index=True)
)

task_evaluator = evaluator("text-classification")

config = AutoConfig.from_pretrained(f"{model_path}config.json")
model = AutoModelForSequenceClassification.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

eval_results = task_evaluator.compute(
    model_or_pipeline=model,
    data=eval_dataset,
    label_mapping={"bacteria": 0, "phage": 1},
    input_column="input_column",
    label_column="label_column",
    tokenizer=tokenizer,
)

print(eval_results)
