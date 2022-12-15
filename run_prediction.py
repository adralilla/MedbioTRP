import os
from tqdm import tqdm
import subprocess

folders = ["Bacteria", "Phages"]
base_folder_path = "./inherit_data/Test/"
outputs_lst = {f.split("/")[-1].split("_out")[0] for f in os.listdir("./outputs/")}
for folder in folders:
    folder_path = f"{base_folder_path}{folder}/"
    input_lst = {f.split("/")[-1].split(".fasta")[0] for f in os.listdir(folder_path)}
    input_lst = input_lst.difference(outputs_lst)
    for fasta_file_name in tqdm(input_lst):
        shell_str = f"python3 IHT_predict.py --sequence {folder_path}{fasta_file_name}.fasta --withpretrain True --model INHERIT.pt --out ./outputs/{fasta_file_name}_out.txt".split(
            " "
        )
        subprocess.run(shell_str)
