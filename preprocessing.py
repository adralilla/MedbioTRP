# import torch
# from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import pandas as pd
# tokenizer = AutoTokenizer.from_pretrained('./models/checkpoint-61500/')
# config = AutoConfig.from_json_file('./models/checkpoint-61500/config.json')
# model = AutoModelForSequenceClassification.from_config(config)
with open("NZ_CP049122.fasta") as f:
    input_seq = f.readlines()[1:]
    
seq=""
for x in input_seq:
    seq += x.replace("\n","")
preprocess = [seq[i:i+6] for i, _ in enumerate(seq) if i<len(seq)-5]
print(len(preprocess))
data = pd.DataFrame({"sequence":[preprocess], "label":"Bacteria"})
data.to_csv("dataset.csv", mode="a", header=False, index=False)
# with open("test3.txt", "w") as f:
#     f.write(str(preprocess))
# tokens = tokenizer(preprocess,padding=True, return_tensors="pt", is_split_into_words=True )
# with open("test.txt", "w") as f:
#     f.write(str(tokens.tokens()))
# with open("test2.txt", "w") as f:
#     f.write(str(tokens.word_ids()))
    
# output = model(**tokens)
# with open ("output.txt", "w") as f:
# 	f.write(output)
 