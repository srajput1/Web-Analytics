
# coding: utf-8

# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import nltk
import re
import string
import csv
from nltk.corpus import stopwords
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import numpy as np



# In[9]:


a=[]
#Opening the file and reading data
with open('final_data2.csv', "r",encoding="Latin1") as f:
    data=csv.reader(f, delimiter=',')
    Date,News,Company,Sentiment,Open,High,Low,Close,Volume=zip(*data)

    
date=list(Date[1:])
news=list(News[1:])
sent=list(Sentiment[1:])
company=list(Company[1:])
open_price=list(Open[1:])
high=list(High[1:])
low=list(Low[1:])
close=list(Close[1:])
volume=list(Volume[1:])
    
    
#     for row[1] in data:
#         a.append(row[1])
sentiment_new=[]
#NLTK Sentiment Analyzer - Vader
sid = SentimentIntensityAnalyzer()
for news1 in news:
    ss = sid.polarity_scores(news1)
    sentiment_new.append(ss)

negative = [val['neg'] for val in sentiment_new]
positive = [val['pos'] for val in sentiment_new]
neutral = [val['neu'] for val in sentiment_new]
compound = [val['compound'] for val in sentiment_new]

#Creating a dataset of the complete data and the sentiment found above
data1 = {'date':date,'news':news,'sentiment_words':sent,'open_price':open_price,'high':high,'low':low,'close':close,'volume':volume,'Neg':negative,'positive':positive,'company':company}
df_senti = pd.DataFrame(data1)
cols = df_senti.columns.tolist()
cols1 = [ 'date','company','news','open_price', 'close', 'high', 'low', 'volume', 'sentiment_words', 'positive','Neg']
df_senti=df_senti[cols1]
df_senti['sentiment_nltk'] = df_senti[['positive','Neg']].idxmax(axis=1)
df_senti['final_sentiment'] = np.where(df_senti['sentiment_nltk']=='positive', 2, 1)
#df_senti.to_csv('final_data_new_sentiment.csv')

#Changing the format of numbers to numeric for corellation to work
df_senti[['open_price','close','high','low','sentiment_words','final_sentiment']] = df_senti[['open_price','close','high','low','sentiment_words','final_sentiment']].apply(pd.to_numeric)

#Changing format of date to plot the sentiments 
df_senti['date'] =  pd.to_datetime(df_senti['date'])

#Getting corellation for all the companies 
list_to_use1 = ['Microsoft','amazon','Boeing','Google','Facebook']
for company in list_to_use1:
    data_use = df_senti[df_senti['company'] == company.lower()]
    print("Corellation for " + company)
    data_use.corr()

    


# In[10]:


pwd


# In[5]:


#Using Groupby to plot the graphs of new found sentiment 
data_grpby = df_senti.groupby(['company','date'])['close','final_sentiment','sentiment_words'].mean()
print(data_grpby)


# In[6]:



#Exporting data to csv
data_grpby.to_csv('new.csv')

with open('new.csv', "r",encoding="Latin1") as f:
    data=csv.reader(f, delimiter=',')

    data1 = list(data)
df1 = pd.DataFrame(data1[1:], index = None, columns=['company','date','close','final_sentiment','sentiment_words'] ,)

#reference:https://matplotlib.org/examples/api/two_scales.html
#getting graph for sentiment using words
list_to_use1 = ['Microsoft','Netflix','amazon','Boeing','Google','Facebook']
for company in list_to_use1:
        
    data_use = df1[df1['company'] == company.lower()]
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(data_use['date'], data_use['close'], 'b-')
    ax1.set_xlabel('Date')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Close_Price', color='b')
    #ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(data_use['date'], data_use['sentiment_words'],'r-')
    ax2.set_ylabel('Sentiment_words', color='r')
    #ax2.tick_params('y', colors='r')

    #fig.tight_layout()
    print(company)
    plt.show()


# In[8]:


#Getting graphs using sentiment taken by Vader package

for company in list_to_use1:
        
    data_use = df1[df1['company'] == company.lower()]
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(data_use['date'], data_use['close'], 'b-')
    ax1.set_xlabel('Date')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Close_Price', color='b')
    #ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(data_use['date'], data_use['final_sentiment'],'r-')
    ax2.set_ylabel('final_sentiment', color='r')
    #ax2.tick_params('y', colors='r')

    #fig.tight_layout()
    print(company)
    plt.show()


# In[7]:


#Getting the graph of both the sentiments for comparision

df1[['sentiment_words','final_sentiment']] = df1[['sentiment_words','final_sentiment']].apply(pd.to_numeric)

df1.groupby('date')['final_sentiment','sentiment_words'].mean().plot(kind='line', figsize=(15,5)).        legend(loc='center left', bbox_to_anchor=(1, 0.5));  # set legend
plt.title('Comparision between Sentiment')
plt.xlabel('Date', fontsize=16)
plt.ylabel('Sentiments', fontsize=16)
plt.show()

