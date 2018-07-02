
# coding: utf-8

# In[112]:
#Spring 2018 AYL6983 - Python for Data Science
#Name: Pratik Jitendra Gala
#This project consists of 3 functions:
#1: Forecasting using Linear Regression
#2: Decompose the time series into its compnents
#3: Forecasting using holtwinters
#Dataset used: AAPL data from yahoo finance (Date: 01/01/2010 - Today)
#Dataset uploaded seperately: Gala_Pratik_AAPL.csv
##Linear Regression
#The below function will take the input parameter as dataframe and will output the predictions for the test data
#It will also give the accuracy for the model


# In[113]:


#Function
def gala_pratik_stockprice_LinearReg(df):
    #Loading the libraries
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    #Splitting the data into train and test
    train=df[0:round(0.75*len(df))]
    test=df[round(0.75*len(df)):]
    #Resting the index
    #train = train.reset_index()
    train_prices = train['Close'].tolist()
    train_dates = train.index.tolist()
    #Reshaping to matrix of n*1
    dates_train = np.reshape(train_dates, (len(train_dates), 1))
    prices_train = np.reshape(train_prices, (len(train_prices), 1))
    test_prices = test['Close'].tolist()
    test_dates = test.index.tolist()
    #Reshaping to matrix of n*1
    dates_test = np.reshape(test_dates, (len(test_dates), 1))
    prices_test = np.reshape(test_prices, (len(test_prices), 1))
    regressor = LinearRegression()
    regressor.fit(dates_train, prices_train)
    plt.figure(figsize=(18,10))
    plt.scatter(dates_test, prices_test, color='red', label = "Actual Price") #plotting the initial datapoints
    plt.plot(dates_test, regressor.predict(dates_test), color='black', linewidth=5, label = "predicted price") #plotting the line made by linear regression
    plt.legend(loc='best')
    accuracy = regressor.score(dates_test, prices_test)
    print("Accuracy of Linear Regression: ", accuracy)
    return(plt.show())


# In[114]:


#User Only needs to load the file - Testing
import pandas as pd
df1 = pd.read_csv('D:\\Desktop\\as\\Topics-python\\Gala_Pratik_AAPL.csv')
gala_pratik_stockprice_LinearReg(df1)


# In[116]:


##Decomposition


# In[117]:


def gala_pratik_stockprice_decomposition(df):
    import pandas as pd
    import statsmodels.api as sm
    #df.set_index(df['Date'], inplace=True)
    decompose=sm.tsa.seasonal_decompose(df.Close,freq = 30)
    return(decompose.plot())


# In[118]:


#Testing (You need to read the dataset and define the index col as Date)
df3 = pd.read_csv('D:\\Desktop\\as\\Topics-python\\Gala_Pratik_AAPL.csv', index_col=['Date'],parse_dates=True)
gala_pratik_stockprice_decomposition(df3)


# In[120]:


##Holt Winter
#Since the time series shows steady seasonal fluctuations, it is additive in nature.
#Hence the value for trend and seasonal will be "add"


# In[121]:


def gala_pratik_stockprice_Holtwinter(df):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import datetime as dt
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    train=df[0:round(0.8*len(df))]
    test=df[round(0.8*len(df)):]
    y = test.copy()
    #Fitting the model
    #Seasonal period = 365 days
    fit1 = ExponentialSmoothing(np.asarray(train['Close']) ,seasonal_periods=365 ,trend='add', seasonal='add',).fit()
    #Forecasting the data
    y['Holt_Winter'] = fit1.forecast(len(test))
    plt.figure(figsize=(18,10))
    plt.plot( train['Close'], label='Train_data')
    plt.plot(test['Close'], label='Test_data')
    plt.plot(y['Holt_Winter'], label='Holt_Winter_prediction')
    plt.legend(loc='best')
    #Root mean square error
    error = sqrt(mean_squared_error(test.Close, y.Holt_Winter))
    print("Root Mean Square Error: ", error)
    return(plt.show())


# In[123]:


#Testing
df2 = pd.read_csv('D:\\Desktop\\as\\Topics-python\\Gala_Pratik_AAPL.csv')
gala_pratik_stockprice_Holtwinter(df2)
