from Bio import Entrez
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import os

dataset_type = "Test"  # Test, Training, Validation
df = pd.read_csv("dataset.csv")
df["Dataset type"].unique()
df_filtered = df[df["Dataset type"] == dataset_type]
df_phage_filtered = df_filtered[df_filtered["Phages/Bacteria"] == "Phages"]
df_bact_filtered = df_filtered[df_filtered["Phages/Bacteria"] == "Bacteria"]

dict_ = {"Phages": {}, "Bacteria": {}}
dict_["Phages"] = set(df_phage_filtered.Accession)
dict_["Bacteria"] = set(df_bact_filtered.Accession)

for type_, accession_lst in dict_.items():
    print(f"Type is: {type_}")
    os.makedirs(f"./data/{dataset_type}/{type_}/", exist_ok=True)
    for accession in tqdm(accession_lst):
        with Entrez.efetch(
            db="nucleotide", id=accession, rettype="fasta", retmode="text"
        ) as handle:
            record = SeqIO.read(handle, "fasta")

        with open(f"./data/{dataset_type}/{type_}/{accession}.fasta", "w") as f:
            SeqIO.write(record, f, "fasta")
