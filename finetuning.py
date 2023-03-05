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

config = AutoConfig.from_pretrained(f"{model_path}config.json")
model = AutoModelForSequenceClassification.from_config(config)

with open(training_set_bact) as f:
    records_train_bact = f.readlines()
records_train_bact = pd.DataFrame({"sentences": random.sample(records_train_bact, n)})

# with open(training_set_phage) as f:
#     records_train_phage = f.readlines()
# records_train_phage = pd.DataFrame(
#     {"sentences": random.sample(records_train_phage, n), "label": 1}
# )

with open(eval_set_bact) as f:
    records_eval_bact = f.readlines()
records_eval_bact = pd.DataFrame({"sentences": random.sample(records_eval_bact, n)})
# with open(eval_set_phage) as f:
#     records_eval_phage = f.readlines()
# records_eval_phage = pd.DataFrame(
#     {"sentences": random.sample(records_eval_phage, n), "label": 1}
# )

train_dataset = Dataset.from_pandas(records_train_bact)

eval_dataset = Dataset.from_pandas(records_eval_bact)

training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
tokenizer = AutoTokenizer.from_pretrained(model_path)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./models/")
