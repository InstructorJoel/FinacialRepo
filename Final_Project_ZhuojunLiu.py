"""
Python Version: 3.6.5

Northeastern University
Spring 2018 AYL6983 - Python for Data Science
Instructor: Joel Schwartz

Capstone Project Financial Team: Apple,Inc Stock price forecasting for 2017
Author: Zhuojun Liu
Date: 06/30/2018

Description: This capstone project aims to use Apple,INC stock price from 2013 to 2016 to create the time series model
forecasting for 2017, and used the 2017 data to versify the model accuracy. At the mean time, compare the different
time series models to find out which model has the highest accuracy. I used sample moving average, sample exponential
moving average, Holt-liner and trend, and Holt-Winter methods for this capstone. The data I used is Apple stock price
from Yahoo Finance with 01/01/2013 to 12/31/2017 time period, and daily frequency.

"""

"""

To test this project, go to the Yahoo Finance website search "AAPL", click "Historical Data", set the time period from 
1/1/2013 to 12/31/2017 and daily frequency, click "Apply" and start download the data (or any other company data with 
same time period and frequency). Read the csv and name it as 'df', set parse_dates = True and index_col='Date'.

Apple data set:
https://finance.yahoo.com/quote/AAPL/history?period1=1357016400&period2=1514696400&interval=1d&filter=history&frequency=1d

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import ExponentialSmoothing


# Def the Root Mean Square Error function
def rmse(x, y):
    """

    :param x: observe data
    :param y: predict data
    :return: Root Mean Square Error
    """
    mse = mean_squared_error(x, y)
    output = pow(mse, 0.5)  # square root the errors
    print('The Root Mean Squared Error is {}'.format(round(output, 3)))  # print the result and round up to 3 digits
    return


# Def the Mean Absolute Percentage Error
def mape(x, y):
    """

    :param x: observe data
    :param y: predict data
    :return: Mean Absolute Percentage Error
    """
    z = abs((x - y) / y)
    output = sum(z) / len(y)
    print('The Mean Absolute Percentage Error is {}%'.format(round(output * 100, 3)))
    print('{}% of data can be explain by your model'.format(round((1 - output) * 100, 3)))
    return


# This function aims for give the user information about how many total missing value in each column, and data types
def summary(x):
    """

    :param x: dateset
    :return: the summary of dataset, include missing value in each column and total number of missing value
    """
    na_info = x.isnull().sum()
    total_na = na_info.sum()
    print(x.head())
    print(x.tail())
    print(x.describe())
    print(x.dtypes)
    print('The missing value in each column shows below:')
    print(na_info)
    print('The total number of missing value is {}'.format(total_na))
    return


# load the Apple stock file and use the 'Date' column as the index number
df = pd.read_csv(r'C:\Users\ckwan\Desktop\AAPL.csv', parse_dates=True, index_col='Date')

# summary the total missing value and data types of each columns
summary(df)

# set up the training set and the testing set
train = df['2013':'2016']
test = df['2017']

# setting the plot styple
plt.style.use('fivethirtyeight')

# plot the training and testing data together
train['Close'].plot(figsize=(15, 8), label='2013 to 2016')
test['Close'].plot(figsize=(15, 8), label='2017')
plt.title('Company Stock Daily Closing Price')
plt.xlabel('Year')
plt.legend()
plt.show()

# Simple Exponential Smoothing Modeling, create the forecasting data set
forecast = test.copy()

# create the column of sample moving average data and plot
forecast['moving_avg'] = train['Close'].rolling(365).mean().iloc[-1]
forecast['moving_avg'].plot(label='Moving Average', figsize=(15, 8))
test['Close'].plot(label='Test', figsize=(15, 8))
train['Close'].plot(label='Train', figsize=(15, 8))
plt.legend(loc='best')
plt.show()

# create the simple exponential smoothing basic on the training data with 0.6 smoothing equation
fit1 = SimpleExpSmoothing(np.asarray(train['Close'])).fit(smoothing_level=0.6)

# forecast the for 2017
forecast['sample_exp_smoothing'] = fit1.forecast(len(test))

# plot the forecasting, training, testing data into one graphic
forecast['sample_exp_smoothing'].plot(label='Simple ExpSmoothing', figsize=(15, 8))
test['Close'].plot(label='Test', figsize=(15, 8))
train['Close'].plot(label='Train', figsize=(15, 8))
plt.legend(loc='best')
plt.show()

# create the seasonal decompose for the training data
sm.tsa.seasonal_decompose(train['Close'], freq=12).plot()
plt.show()

# generate the Holt Linear & Trend modeling
fit2 = Holt(np.asarray(train['Close'])).fit(smoothing_level=0.3, smoothing_slope=0.1)

# use the Holt linear $ Trend model to forecast the 2017
forecast['holt_linear'] = fit2.forecast(len(test))

# compare the trend and the actual data
forecast['holt_linear'].plot(label='Holt Linear&Trend', figsize=(15, 8))
test['Close'].plot(label='Test', figsize=(15, 8))
train['Close'].plot(label='Train', figsize=(15, 8))
plt.legend(loc='best')
plt.show()

# create the Holt Winter model with daily period for 2017
fit3 = ExponentialSmoothing(np.asarray(train['Close']), trend='add', seasonal_periods=365, seasonal='add').fit()

# forecast the Holt Winter for 2017
forecast['holt_winter'] = fit3.forecast(len(test))

# plot and compare the Holt Winter, training and testing data, and the plot will take a while to generate
forecast['holt_winter'].plot(label='Holt Winter', figsize=(15, 8))
test['Close'].plot(label='Test', figsize=(15, 8))
train['Close'].plot(label='Train', figsize=(15, 8))
plt.legend(loc='best')
plt.show()

# calculate root mean square error for each model
rmse(test['Close'], forecast['moving_avg'])
rmse(test['Close'], forecast['sample_exp_smoothing'])
rmse(test['Close'], forecast['holt_linear'])
rmse(test['Close'], forecast['holt_winter'])

# calculate the mean absolute percentage error for each model
mape(test['Close'], forecast['moving_avg'])
mape(test['Close'], forecast['sample_exp_smoothing'])
mape(test['Close'], forecast['holt_linear'])
mape(test['Close'], forecast['holt_winter'])

"""

The Holt-Winter model needs about 30 seconds to generate, and the RMSEs and MAPEs will calculate after the Holt-Winter 
is created.

Result: compare the each model's RMSE and MAPE, I find out the Holt-liner and trend, and the Holt-Winter model 
have the lowest root mean square error and lowest mean absolute percentage error. 

"""
