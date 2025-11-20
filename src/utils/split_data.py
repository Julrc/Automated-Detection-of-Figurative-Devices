#%%
import pandas as pd

class ReadTSV:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def read_tsv(self):
        self.df = pd.read_csv(self.file_path, sep='\t')
        return self.df

    def split_data(self, split_index=2160):
        if self.df is None:
            self.read_tsv()
        training_df = self.df.iloc[:split_index]
        test_df = self.df.iloc[split_index:]
        return training_df, test_df

# %%
