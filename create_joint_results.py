import pandas as pd
import os
import numpy as np

output_df = pd.DataFrame(columns=["name", "category", "score"])

for filename in os.listdir("./outputs/"):
    df = pd.read_csv(f"./outputs/{filename}", sep="\t", header=0)
    if not df.empty:
        output_df = pd.concat([output_df, df], ignore_index=True)

output_df["name"] = output_df["name"].str.split(".").str[0]
df = pd.read_csv("paper_results.csv")
df["name"] = df["name"].str.split(".").str[0]

result_df = df.merge(output_df, how="outer", on="name")

gt_df = pd.read_csv("dataset.csv")
gt_df = gt_df[gt_df["Dataset type"] == "Test"]
gt_df = gt_df.reset_index(drop=True)
gt_df = gt_df.drop(["Dataset type"], axis=1)
gt_df = gt_df.rename(columns={"Accession": "name"})

result_GT_df = result_df.merge(gt_df, how="outer", on="name")
result_GT_df = result_GT_df.rename(columns={"Phages/Bacteria": "Ground_truth"})
result_GT_df.to_csv("outputs_with_GT.csv", index=False)
