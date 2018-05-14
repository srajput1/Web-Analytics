
# coding: utf-8

# In[115]:


#Sourabh Rajput
#BIA 660-C analytics 
#CWID 10431188


import numpy as np
from pandas import DataFrame
import pandas as pd
import csv

def analyze_tf(arr): #takes array as an input  
    
    print("*****Question 1*******\n")
    
    tf_idf=None
    #print(arr)
    
    a1=np.sum(arr,axis=1) #for normalizing friquency selects each elements  
    #print(a1,"\n")
    
    tf= np.divide(arr.T,a1) #decides word frequency by length of the document and saves result in tf 
    #print(tf.T)
    
    a2=np.where(tf.T>0,1,0)
    #print(a2)
    
    df=np.sum(a2,axis=0) #calculates the document frequency  of each word in df
    #print(df)
    
    tf_idf=tf.T/df #tf/df 
    #print(tf_idf)
    
    f=np.argsort(tf_idf)# it will count the indexs according to word which was appered most of the time in ascending order
    print(f[:,-3:]) # prints only indexes of words with top 3 largest values in the tf_idf array
    
    
    return tf_idf

def analyze_cars():
    
    print("\n********Question 2 ********\n")
    ReadFile=pd.read_csv('cars.csv') # reads file cars.csv
    #print(ReadFile)
    #data1=ReadFile.readlines()
    
    df=ReadFile.sort_values(by=['cylinders','mpg'],ascending=False) #sorts in descending order 
    tf=df.head(3) # will print only first 3 rows
    print(tf)
    
    df['brand']=df.apply(lambda row:row["car"].split(' ')[0], axis=1) # create column brand and from car column selects first word and print it
    print("\n",df)
    
    tf=df[df.brand.isin(['ford','buick','honda'])].groupby(["cylinders","brand"]) #Considers only for ford buick and honda brands on cylinders value 
    tf1=tf['acceleration'].agg([np.mean, np.min, np.max]) #finds min max and mean 
    print(tf1)
    
    idf=pd.crosstab(index=df.brand,columns=df.cylinders, values=df.mpg,aggfunc=np.mean)
    print(idf)
    
    #return tf_idf

if __name__ == "__main__":  
    
    #1 Test Question 1
    arr=np.random.randint(0,3,(4,8))

    tf_idf=analyze_tf(arr)
    
   # Test Question 2
    analyze_cars()

