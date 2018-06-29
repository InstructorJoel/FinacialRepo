import pandas as pd
import numpy as np


def read_data(csv_file):
    # type: (csv_file) -> df
    df = pd.read_csv(csv_file)
    return df


def tokenize(df):
    # count numbers and store in df
    df = df.assign(world_count=0, number_count=0)
    df['world_count'] = df['test'].apply(lambda x: len(str(x).split(" ")))
    df['number_count'] = df['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    return df


def liner_regression(attribute1, attribute2, df):
    from sklearn import linear_model
    # Create linear regression object
    regr = linear_model.LinearRegression()
    X = df[attribute1]
    Y = df[attribute2]
    regr.fit(X, Y)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    return


def visualization(attribute1, attribute2, df):
    import seaborn as sns
    sns.set(color_codes=True)
    np.random.seed(sum(map(ord, "regression")))
    sns.lmplot(x=attribute1, y=attribute2, data=df)
    return


if __name__ == '__main__':
    # https://www.kaggle.com/yelp-dataset/yelp-dataset/home
    df_review = read_data('C:\Users\star\Desktop\yelp_review.csv')
    df = tokenize(df_review)
    liner_regression('world_count', 'stars',df)
    visualization('world_count', 'stars',df)
    liner_regression('world_count', 'useful', df)
    visualization('world_count',  'useful', df)
    liner_regression('world_count', 'funny', df)
    visualization('world_count', 'funny', df)
    liner_regression('world_count', 'cool', df)
    visualization('world_count', 'cool', df)
    liner_regression('number_count', 'stars', df)
    visualization('number_count', 'stars', df)
    liner_regression('number_count', 'useful', df)
    visualization('number_count', 'useful', df)
    liner_regression('number_count', 'funny', df)
    visualization('number_count', 'funny', df)
    liner_regression('number_count', 'cool', df)
    visualization('number_count', 'cool', df)





