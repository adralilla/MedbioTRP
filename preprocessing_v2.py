import pandas as pd
import os

dataset_name = "./train_dataset.csv"

pd.DataFrame(columns=["sequence", "label"]).to_csv(dataset_name, mode="w", index=False)

folder_set = {"Bacteria": "Bacteria", "Phages": "Phage"}
for data_type, label in folder_set.items():
    folder = f"./inherit_data/data_train/Training/{data_type}/"
    for filename in os.listdir(folder):
        if ".fasta" in filename:
            with open(folder + filename) as f:
                input_seq = f.readlines()[1:]

            seq = ""
            for x in input_seq:
                seq += x.replace("\n", "")

            preprocess = [seq[i : i + 6] for i, _ in enumerate(seq) if i < len(seq) - 5]

            data = pd.DataFrame({"sequence": [preprocess], "label": label})

            data.to_csv(dataset_name, mode="a", header=False, index=False)
