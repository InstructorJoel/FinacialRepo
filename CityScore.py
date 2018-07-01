
# coding: utf-8

# In[1]:


'''
Topic - City Score summary and prediction with Boston Data
Author - Sunita Mohapatra
Date - 29/06/2018
Description - City Score contains multiple data including BFD response time, crime score, homicides, etc. I have tried to use multiple existing
libraries to generate a clean report and I have the basic OOPs concept of python including the constructor method to pass the data 
between the functions inside a class. I have tried to give more focus on the visualization part for making it more user friendly.


'''
import pandas as pd
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA 
from pandas.tools.plotting import autocorrelation_plot
from tqdm import tqdm # progress bar
from sklearn.metrics import mean_squared_error
#Install the missing packages by: python -m pip install <package name> or pip install <package name>
class CSPredict:
    def __init__(self,path): 
        self.path=path
        
    def parser(self,x): #Returns the date/already present in excel.
        return x
    def start(self): #reads the excel, plots graph wrt dats and score, calls statistics functions
        
        #self is defined in constructor and can be accessed/changed in all the functions inside particular class.
        # read and parse data 
        print("#############################################################")
        print("###################CITY SCORE PREDICTION #############")
        print("#############################################################")
        print("-------------------------------------------------------------")
        city_score = read_csv(self.path, header=None,parse_dates=[2], squeeze=True, date_parser=self.parser)
        city_score = city_score.iloc[:, [2, 1]]
        city_score.columns = ['Date', 'Score']

        # set index as datetime to support plotting and Arima Model
        city_score.set_index('Date', inplace=True)
        # city_score = city_score.asfreq('T')

        print(city_score.head())

        city_score.plot(title='City Score Prediction')
        pyplot.tight_layout()
        pyplot.show()
        print("Printing the statistics")
        print("-----------------------------------------")
        self.stats(city_score) #stats function is called here with the parameter cityscore as an object with the value of data and score
    def stats(self,city_score): #shows the total statistics with imaginary numbers
        model = ARIMA(city_score, order=(5,1,0))
        model_fit = model.fit(disp=0)
        print(model_fit.summary())

        # plot residual errors
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        pyplot.show()
        residuals.plot(kind='kde')
        pyplot.show()
        print(residuals.describe())
        print("-----------------------------------------")
        print("Please wait...... Training the testcases is in progress...")
        print("-----------------------------------------")
        self.autocorelaton(city_score)
    def autocorelaton(self,city_score): #data is trained
        
        #
        autocorrelation_plot(city_score)
        pyplot.show()
        #Will take more than 30 minutes to 1 hour to run as it will train the data.
        X = city_score.values
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)] #train starts from 0 size, trains the memory size 
        history = [x for x in train]
        predictions = list()
        print('Total Test cases: {}'.format(len(test)))
        for t in tqdm(range(len(test))):
            model = ARIMA(history, order=(5,1,0)) #5 for autoregression - 1 for time series stationary 
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat) #prediction data(all the data are stored in yhat and the group formed will be stored in prediction list) )
            obs = test[t]
            history.append(obs)
        self.finalize(test,predictions)
    def finalize(self,test,predictions): #finalize checks the error and shows the graph with prediction
        error = mean_squared_error(test, predictions)
        print('Test: %.3f' % error)
        # plot
        pyplot.plot(test)
        pyplot.plot(predictions, color='red')
        pyplot.show()
        print("------------------------------------------")
        print("END")

#CSPredict('cityscore.csv').start(); For running, implement this statement

