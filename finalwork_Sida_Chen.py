import panda as pd
import numpy as np


def read_data(csv_file):
    # type: (csv_file) -> df
    df = pd.read_csv(csv_file)
    return df


def tokenize(df):
    df['world_count'] = df['test'].apply(lambda x: len(str(x).split(" ")))
    df['number_count'] = df['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    return df


def liner_regression(X, Y):
    import matplotlib.pyplot as plt
    from sklearn import linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X, Y)
    # Make predictions using the testing set
    Y_pred = regr.predict(X)
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(Y, Y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(Y, Y_pred))
    plt.scatter(X, Y, color='black')
    plt.plot(X, Y_pred, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    return


if __name__ == '__main__':
    df_business = read_data('C:\Users\star\Desktop\yelp_business.csv')
    df_review = read_data('C:\Users\star\Desktop\yelp_review.csv')
    tokenize(df_review)



