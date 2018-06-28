import panda as pd
import numpy as np
def read_data(csv_file):
    df = pd.read_csv(csv_file)
    return df
def tokenize(df):
    for i in df.nrows:
        df['world_count']=df[i,'text'].apply(lambda x: len(str(x).split(" ")))
        df['number_count'] = df[i,'text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    return df
def liner_regression(df):
    import statsmodels.api as sm
    target =

if __name__ == '__main__':
