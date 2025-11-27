#%%
from utils.split_data import ReadTSV
from transformers import AutoTokenizer
#%%

annotation_path = '../data/nlp_annots.tsv'
data = ReadTSV(annotation_path)

training_df, test_df= data.split_data()
