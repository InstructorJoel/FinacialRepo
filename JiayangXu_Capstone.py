
"""
ALY6983 ST: Python for Data Science

Capstone Project-Financial team

Northeastern University 

Pro.Joel Schwartz

Author: Jiayang Xu

07/02/2018

Python Version: 3.6

"""


"""
My function has two parts. I intend to help users to view the stock price. They can define the ticker,
the start date, and end date. By observing the movement of stock price, users enable to conduct further 
analysis. In technical analysis,it is important to observe the stock's moving average. I define the 
function on the Moving Average. The users can choose which moving average to observe. Besides, I use the
logistic regression to calculate the stock return.Users can define the different variables to view the 
return of stocks. 
"""


#Import the library
import numpy as np
import pandas as pd
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import talib as tb
from datetime import date
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


####Part1####


"""
I use XiaoLiu's function to get the ticker of the company. The users can input the 
company name to get the ticker. And I will use the ticker to load data. 
"""
from Final_Project_XiaoLiu import get_company
from Final_Project_XiaoLiu import get_ticker
from Final_Project_XiaoLiu import company_ticker

def get_company():
   company_input=input("Please enter a company name:")
   return company_input

import requests 
def get_ticker(name):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(name)

    result = requests.get(url).json()

    for x in result['ResultSet']['Result']:
        if x['name'] == name:
            return x['symbol']

def company_ticker():
    while True:
        try:  
            ticker_get=get_ticker(get_company()) #There is a nested function which asks users to put a company name and then return a stock ticker
            ticker_get1=ticker_get[:4]
            return ticker_get1
        except: 
            print("This company is not public!")# If the company is not public, then an error message returns and user needs to put another name
            continue
        else:
            break

print(company_ticker()) #Input the Amazon.com and get the ticker

"""
Load the financial data from Yahoo Finance. After the users can input the company name and get the 
ticker. And I define the function which users are able to input the ticker,start date, and the end 
date to get the financial data. I also define some indicators and users enable to check the moving 
average by using this function.
"""

#Define the function: Load the stock price from Yahoo.finance 
def load_data(code,start_date,end_date):
    sp=yf.download(code,start_date,end_date)
    return sp

#Input the ticker/start date/end date and get the stock price. I take the AMAZ for example. 
sp=load_data('AMZN','2010-01-01','2018-01-01') #users can change the ticker and start date and end date.
sp=sp.dropna()
print(sp)

#Define the function: Calculate the moveing average 
def MA(timeperiod):
    MA=tb.SMA(sp.Close,timeperiod)
    return MA

#Get the short-term moving average and long-term moving average 
sp["MAShort"]=MA(50)
sp["MALong"]=MA(200)

#Plot the 50-day moving average and plot the 200-day moving average
def show_MA():
    plt.figure(figsize=(10,5))
    plt.plot(sp.Close)
    plt.plot(sp.MAShort,label='MA50')
    plt.plot(sp.MALong,label='MA200')
    plt.grid()
    plt.legend()

show_MA()


####Part2####


"""
I plan to conduct the logistic regression to create a trading strategy. I intend to define the 50-day 
moving average, correlation, relative strength index(RSI), the difference between the open price of 
yesterday and today, and difference close price of yesterday and open price of today as the independent 
variables. The core of the strategy is defining the dependent variable. When the tomorrow's closing price 
is greater than the today's closing price,I choose to buy the stock. This action is defined as 1, 
When the tomorrow's closing price is less than the today's closing price, I choose to sell the stock. This
action is defined as -1.
"""

#Load the stock data
stock=sp.dropna()
stock=stock.iloc[:,:4]
stock.head()

#Define dependent variables 
def spclosecorr(timeperiod):
    closecorr=stock["Close"].rolling(timeperiod).corr(MA(timeperiod))
    return closecorr
def spRSI(timeperiod):
    spRSI= tb.RSI(np.array(stock['Close']),timeperiod)
    return spRSI

#Choose the 50-day moving average, correlation, and the relative storing indicators
MA50=MA(50) #Users can choose other moveing average 
cc=spclosecorr(50)
rsi=spRSI(50)

#Define the independent variables
stock["50-day Moving average"]=MA50
stock["Correlation"]=cc
stock["RSI"]=rsi
stock['Open-Close'] = sp['Open'] - sp['Close'].shift(1)
stock['Open-Open'] = sp['Open'] - sp['Open'].shift(1)
stock.head()
stock=stock.dropna()
s=stock.iloc[:,:9]

#define the target variable
target = np.where (stock['Close'].shift(-1) > stock['Close'],1,-1)

#Split the dataset
split_data = int(0.8*len(stock))
Ind_train,Ind_test, Tar_train, Tar_test = s[:split_data],s[split_data:],target[:split_data],target[split_data:]

#Build the logistic regression model 
logisticmodel=LogisticRegression()
logisticmodel=logisticmodel.fit(Ind_train,Tar_train)

#Define the function: calculate the return of stock and strategy return
def model_return():
  stock['predict_price'] = logisticmodel.predict(s)
  stock['stock_returns'] = np.log(stock['Close']/stock['Close'].shift(1))
  cum_return = np.cumsum(stock[split_data:]['stock_returns'])  #Calculate the return of stock
  stock['Startegy_returns'] = stock['stock_returns']* stock['predict_price'].shift(1) #Calculate the strategy return of stock
  cum_strategyreturns=np.cumsum(stock[split_data:]['Startegy_returns'])
  plt.figure(figsize=(10,7))
  plt.plot(cum_return, color='yellow',label = 'Returns')
  plt.plot(cum_strategyreturns, color='red', label = 'Strategy Returns')
  plt.legend()

model_return()

"""
I take the Amazon for example. Users can choose different stock and define different independent variables to get the return and strategy return.
"""
