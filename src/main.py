#%%
from utils.split_data import ReadTSV

annotation_path = 'data/annots.tsv'
data = ReadTSV(annotation_path)

training_df, test_df= data.split_data()
# %%

