
"""

ALY6983 ST: Python for Data Science
Capstone Project

Northeastern University 
Instructor: Joel Schwartz

Author:Xiao Liu
June 2018
Financial Group

Python Version: 3.6
"""


"""
This project aims to help users to find company tickers and prices as well as company news at their interest. 
With several input/output functions, the project could return a dataframe with stock price and price difference perentage change. 
If users could visualize a high movement in the price data frame, they could easily find relative company news with company name. 
"""
################################Part I#########################################
"""
This part is a preparation for the next which grabs company ticker and date to find relative stock price
"""

def get_company():
   """This function asks the user to put an interested company name with input function
   """
   company_input=input("Please enter a company name:")
   return company_input


import requests 
def get_ticker(name):
    """
        This function could easily grab company ticker with company name with a useful API
    """
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(name)

    result = requests.get(url).json()

    for x in result['ResultSet']['Result']:
        if x['name'] == name:
            return x['symbol']


def company_ticker():
    """
    This function retrieves company ticker with company name, if the company is not public, an error message returns
    """
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
            

####################################PartII#################################### 
"""
This part take the function and parameters obtained from part one and find the real price
"""

from datetime import datetime #Use this library to get date and time features package in python
def get_date():
    """ 
        This function could ask user to input a date of which he/she is interested in finding the stock price
    """
    date_input=input('Please enter a date in format YYYY-MM-DD: ')
    year,month,day=map(int,date_input.split('-'))
    date=datetime(year,month,day)
    return date


import pandas_datareader.data as web #pandas_datareader could easily grab web data 
def data_value():
    """
    This function could get a specific stock price on date which user is interested
    """
    start=get_date() #This is a nested function which asks user to put in a date, if user has a date in mind, this date could be just a date rather than an input
    end=start
    stock_price=web.DataReader(company_ticker(), 'morningstar',start,end)#The source of stock price in this case is "morningstar". Yahoo finance and google finance could be used as well but he source is not as stable
    return stock_price
print(data_value())


"""
This part asks user how many days of prices they would like to see and then stock price could be used to compare which is the actual part to run
"""
import pandas as pd
data_frame=pd.DataFrame()
while True:
    try:
        number_of_prices=int(input("How many prices you would like to see: "))#Asks user to input an integer and if it is not an integer, an error message return and user needs to input another one
        
    except ValueError:
        print("Please enter an integer!")
        continue
    else:
        break
i=1
while i<=number_of_prices:
    data_frame=data_frame.append(data_value())
    i=i+1
data_frame.reset_index(level=0,inplace=True)
print(data_frame)#A data frame with several days' prices is built


#####################################Part III##################################
"""
This part calculates the price movement of prices obtained from part I and II.
"""

data_frame['Close Price Percentage Change']=abs(data_frame['Close'].pct_change())# Price percentage change is calculated
data_frame
list(data_frame)

#Check if there is a movement that is more than 10% in the price percentage change
for i in data_frame['Close Price Percentage Change']:
    if (i>=0.1 or i<=-0.1):
        print('There is a high movement!')
    else:
        print('There is no significant movement!')

#####################################Part IV####################################
"""
For this last part, I used Huidan Zhang's web parsing function which could return 
interested company news. From last part, if there is a high price movement, users might 
be interested in discovering if there is somehting important happened to the company. 
News headlines are crucial resources to find such information. And if they are interested in 
digging detailed news, they could further search the full text of the article. 
"""

from bs4 import BeautifulSoup
from capstone_huidan_zhang import parse_html
company=get_company()
url= 'https://www.bloomberg.com/search?query='+company
variable=parse_html(url)
variable2=variable.find_all('div',{'class':'search-result-story__container'})
for i in variable2:
    article=i.text
    print(article)

"""
Testing: I will use "Apple" as an example to show how the whole project works
"""













