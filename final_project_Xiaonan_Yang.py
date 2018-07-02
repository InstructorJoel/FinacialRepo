
# coding: utf-8

# In[9]:


# In the past year, Bitcoin has been fluctuated frequently - the highest market price was over USD19,000, the lowest market price was around USD2,000, and there was approximately 6 main fluctuations. In this project, my goal is to find out when the best timing is to buy Bitcoin. To achieve this goal, the mean value of Bitcoin's market price and snapped price will be viuslized. The dataset is download from coingecko.com
import pandas as pd


# In[17]:


#import the data set
data = pd.read_csv(r'C:\Users\seany\OneDrive\Documents\btc_usd.csv')
#create a dataframe of this dataset
df = pd.DataFrame(data)
df


# In[23]:


#two of clomuns in this dataset is useless for this project, so I deleted them from the dataframe
df.drop(columns=['market_cap','total_volume'])


# In[39]:


#change the index of this dataset from number to the date
df.index = df['snapped_at']
df.head


# In[27]:


#create the plot of the price change
get_ipython().run_line_magic('pylab', 'inline')
df['price'].plot(kind = 'line', figsize = [10,6])


# In[29]:


#find the mean value at each week, and name the mean value as "ma7"
df['ma7'] = df['price'].rolling(window = 7).mean()
df['ma7'].head


# In[34]:


#plot the whole graph of the market price and weekly average market price
df[['price','ma7']].plot(kind = 'line',figsize =[20,10])


# In[35]:


#find out last month's market price vs average price
df2 = df[df['snapped_at'] >= '2018-06-03 00:00:00 UTC']


# In[36]:


df2.head


# In[37]:


#plot last month's market price and average price, so that I can know the best timing to buy the Bitcoin
df2[['price','ma7']].plot(kind = 'line',figsize =[20,10])

