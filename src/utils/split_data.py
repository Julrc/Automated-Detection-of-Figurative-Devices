#%%
import pandas as pd

class ReadTSV:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def read_tsv(self):
        self.df = pd.read_csv(self.file_path, sep='\t')

        self.df = pd.read_csv(self.file_path, sep="\t", header=None, dtype=str)
        self.df.columns = ["text", "device", "rhyme"]

        return self.df

    def split_data(self, split_index=2160):
        if self.df is None:
            self.read_tsv()
        training_df = self.df.iloc[:split_index]
        test_df = self.df.iloc[split_index:]
        test_df = test_df.reset_index(drop=True)

        columns = ["device", "rhyme"]

        for col in columns :
            training_df[col] = training_df[col].fillna("literal")
            test_df[col] = test_df[col].fillna("literal")

            training_df[col] = training_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)
            training_df[col] = training_df[col].str.lower().str.strip().str.split(',').str[0].str.strip()
            test_df[col] = test_df[col].str.lower().str.strip().str.split(',').str[0].str.strip()

        return training_df, test_df

# %%
