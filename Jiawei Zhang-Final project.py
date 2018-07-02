## ALY6983 ST: Python for Data Science
## Capstone Project
## Northeastern University 
## Instructor: Joel Schwartz
## Author:Jiawei Zhang
## June 2018
## Group
## Python Version: 3.6


## This project aims to help ISOs or the market participates to understand the load trend better,
## take full advantage of the market information, and increase their market competitive capability. 
## Several time series models have been adopted in this project to analyze the SPP load trend and forecast the future load consumption. 
## For each method, the RMS, Root mean square, has been calculated, which used to check the performance of each method. 
## SO far, based on the current data we have, the method 2, Simple Average, did the best.  



## the data came from: https://marketplace.spp.org/pages/hourly-load

import pandas as pd # import the pandas tool box
import numpy as np 	# import the numpy tool box
import matplotlib.pyplot as plt # import the matplotlib tool box 
from sklearn.metrics import mean_squared_error # import the sklearn tool box
from math import sqrt # import the math tool box
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt # import the statsmodels tool box
import statsmodels.api as sm # import the statsmodels tool box

# define a fucntion, in which will be used to aggregate hourly data into daliy averge data
def Aggregating(x):
	x.Timestamp = pd.to_datetime(x.MarketHour) 
	x.index = x.Timestamp 
	return x.resample('D').mean()

# define a plot function, in which will be used to plot, train data, test data, and predicted data.
def draw(x,y,z,name):
	plt.plot(x, 'red', y, 'green', z, 'blue')
	plt.xlabel('Time')           # X-axis label
	plt.ylabel('Load')              # Y-axis label
	plt.title(name) # Overall title
	plt.legend(['train', 'test', 'predicted'], loc = 'best')
	plt.show()
	
# Importing data: SPP hourly load 2015/01/01-2018/05/31
df = pd.read_csv('train.csv')

# Creating train dataset, which used to build the model and test dataset, which used to test the forecast performance 

train = df[0:26296]  # Choose data of from 2015/01/01 to 2017/12/31 as training dataset
test  = df[26297:]   # Choose data of from 2018/01/01 to 2018/05/31 as testing dataset


# Aggregating the dataset from hourly load into daily average load
train = Aggregating(train) #  
test  = Aggregating(test)  #

# plot the SPP load as daily average, in which the red line represent the trainning dataset and the green line represent the test dataset 
plt.plot(train, 'red', test, 'green')
plt.xlabel('Time')           # X-axis label
plt.ylabel('Load')              # Y-axis label
plt.title('Time series of SPP Load') # Overall title
plt.legend(['train', 'test'])
plt.show()

# Method 1: Assume that the load of hour t, equals to the load of the next hour, t+1
# the equations could be represent by Load(the next time) = Load (the current time)
dd= np.asarray(train['CSWS'])
predicted1 = test.copy()
predicted1['CSWS'] = dd[len(dd)-1]

# plot the SPP load as daily average, 
# in which the red line represents the trainning dataset,
# the green line represents the test dataset, and the blue line represents the forecasted value 
name = 'method 1'
draw(train,test,predicted1,name)

# calculate the RMS of the method 1, and check the performance of method 1
rms_Method1 = sqrt(mean_squared_error(test['CSWS'], predicted1['CSWS']))
print("rms_Method1:",rms_Method1)

# Method 2: Simple Average, which assume that the load of the next time, equals to the average load of the past.
# the equations could be represent by Load(the next time) = average of the past Load.

predicted2 = test.copy()
a = np.asarray(train['CSWS'])
predicted2['CSWS'][0] = a.mean()

for i in range(test.shape[0]-1):	
	a = np.append(a,predicted2['CSWS'][i])
	predicted2['CSWS'][i+1] = a.mean()  
	
# plot the SPP load as daily average, 
# in which the red line represents the trainning dataset,
# the green line represents the test dataset, and the blue line represents the forecasted value 
name = 'method 2'
draw(train,test,predicted2,name)
rms_Method2 = sqrt(mean_squared_error(test['CSWS'], predicted2['CSWS']))
print("rms_Method2:",rms_Method2)


# Method 3: Moving Average, which assume that the load of the next time, equals to the average load of the past pre-setting days.
# the equations could be represent by 
# Load(t) = [ Y(t-1) + Y(t-2) + ... + Y(t-p) ] / p

P = 20; 
predicted3 = test.copy()
predicted3['CSWS'] = train['CSWS'].rolling(P).mean().iloc[-1]

# plot the SPP load as daily average, 
# in which the red line represents the trainning dataset,
# the green line represents the test dataset, and the blue line represents the forecasted value 
name = 'method 3'
draw(train,test,predicted3,name)
rms_Method3 = sqrt(mean_squared_error(test['CSWS'], predicted3['CSWS']))
print("rms_Method3:",rms_Method3)


# Method 4: Simple Exponential Smoothing: use the Exponential function to traning the past load.
# the equations could be represent by 
# Load(t) = Exponential^(a*Load(t-1))

predicted4 = test.copy()
alpha = 0.1
fit = SimpleExpSmoothing(np.asarray(train['CSWS'])).fit(smoothing_level=alpha,optimized=False)
predicted4['CSWS'] = fit.forecast(len(test))

# plot the SPP load as daily average, 
# in which the red line represents the trainning dataset,
# the green line represents the test dataset, and the blue line represents the forecasted value 
name = 'method 4'
draw(train,test,predicted4,name)
rms_Method4 = sqrt(mean_squared_error(test['CSWS'], predicted4['CSWS']))
print("rms_Method4:",rms_Method4)

# Method 5: Holt,s Linear Trend method
sm.tsa.seasonal_decompose(train['CSWS']).plot()
result = sm.tsa.stattools.adfuller(train['CSWS'])
plt.show()
# Each Time series dataset can be decomposed into it,s componenets 
# which are Trend, Seasonality and Residual. 
# Any dataset that follows a trend can use Holt,s linear trend method for forecasting.

predicted5 = test.copy()
alpha = 0.1
lamda = 0.1
fit1 = Holt(np.asarray(train['CSWS'])).fit(smoothing_level = alpha,smoothing_slope = lamda)
predicted5['CSWS'] = fit1.forecast(len(test))

# plot the SPP load as daily average, 
# in which the red line represents the trainning dataset,
# the green line represents the test dataset, and the blue line represents the forecasted value 
name = 'method 5'
draw(train,test,predicted5,name)
rms_Method5 = sqrt(mean_squared_error(test['CSWS'], predicted5['CSWS']))
print("rms_Method5:",rms_Method5)

# Method 6: Holt Winters Method
predicted6 = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['CSWS']) ,seasonal_periods=7 ,trend='add', seasonal='add',).fit()
predicted6['CSWS'] = fit1.forecast(len(test))
# Using Holts winter method will be the best option among the rest of the models beacuse of the seasonality factor. 
# The Holt-Winters seasonal method comprises the forecast equation and three smoothing equations: 
# one for the level t, one for trend t and one for the seasonal component denoted by st, with smoothing parameters.

# plot the SPP load as daily average, 
# in which the red line represents the trainning dataset,
# the green line represents the test dataset, and the blue line represents the forecasted value 
name = 'method 6'
draw(train,test,predicted6,name)
rms_Method6 = sqrt(mean_squared_error(test['CSWS'], predicted6['CSWS']))
print("rms_Method6:",rms_Method6)


# Method 7: ARIMA:# Method 
# Autoregressive Integrated Moving average. 
# ARIMA models aim to describe the correlations in the data with each other. 
predicted7 = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train['CSWS'], order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
predicted7['CSWS'] = fit1.predict(start="2018-1-1", end="2018-5-31", dynamic=True)

# plot the SPP load as daily average, 
# in which the red line represents the trainning dataset,
# the green line represents the test dataset, and the blue line represents the forecasted value 
name = 'method 7'
draw(train,test,predicted7,name)
rms_Method7 = sqrt(mean_squared_error(test['CSWS'], predicted6['CSWS']))
print("rms_Method7:",rms_Method7)

