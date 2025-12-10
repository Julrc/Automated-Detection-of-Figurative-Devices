#%%
import pandas as pd
from utils.clean_data import ReadTSV
import re

#%%

annotation_path = '../data/nlp_annots.tsv'
data = ReadTSV(annotation_path)

df = data.read_tsv()

task = "simile"
df.loc[df['device'] != task, 'device']= 'literal'
#%%

device_regex = r"\b(like|as)\b"

false_positive=0
true_positive=0
true_negative=0
false_negative=0

#%%
for index, row in df.iterrows():

    if (re.search(device_regex, row['text']) is not None):
        if (row['device'] == task):
            true_positive+=1
        else:
             false_positive += 1
    else:
        if (row['device'] == task):
            false_negative+=1
        else:
            true_negative+=1
        

# %%
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
recall = true_positive / (true_positive + false_negative)
precision = true_positive / (true_positive + false_positive)
f1 = (2 * precision * recall) / (precision + recall)

# %%
print(f"Accuracy: {accuracy * 100}%")
print(f"Recall: {recall* 100}%")
print(f"Precision: {precision *100}%")
print(f"F1: {f1}%")

# %%
