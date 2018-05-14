
# coding: utf-8

# In[157]:


#Sourabh Rajput
#BIA 660 C
#CWID 10431188


import pandas as pd
import requests 
from bs4 import BeautifulSoup  
from pandas import DataFrame
import numpy as np 
import matplotlib.pyplot as plt


def mpg_plot():
    
    df = pd.read_csv('auto-mpg.csv', header=0) # reads file auto-mpg.csv
    #print(df)
    #df.head()
    #df.info()
    ctab=pd.crosstab(index=df.model_year,columns=df.origin,values=df.mpg, aggfunc=np.mean)
    
    ctab.plot(kind='line',figsize=(8,4), title="avg.mpg by origin over years ").legend(loc='center left', bbox_to_anchor=(1, 0.5));
   
    plt.show()
    
def getReviews(movie_id):        
    reviews=[]  # variable to hold all reviews 
    
    page_url="https://www.rottentomatoes.com/m/"+movie_id+"/reviews/?type=top_critics"
    
    page = requests.get(page_url)         
    soup = BeautifulSoup(page.content, 'html.parser')  
    if page.status_code==200:  
            
        SelectData=soup.select("div#reviews div.row.review_table_row")
            # insert your code to process page content                          
        for idx, SelectData in enumerate(SelectData):
           
            reviewer=SelectData.select("div div.critic_name a") # to get Reviewer Name
            
            if reviewer!=[]:
                reviewer=reviewer[0].get_text()

                                                    
            Date=SelectData.select("div.review_date") # get Review Date
            if Date!=[]:
                Date=Date[0].get_text()

           
            Description=SelectData.select("div.review_desc div.the_review")  # get Review Description
            if Description!=[]:
                Description=Description[0].get_text()

           
            score=SelectData.select("div.review_desc div.small.subtle")  # get Original Score
            if score!=[]:
                score1=score[0].get_text()[31:]
                #print(len(score1))
                #print(type(score1))
                if score1:
                    pass
                else:
                    score1="N/A"
                
                
            # add Reviewer Name, Review Date, Review Description, and Original Score as a tuple into the list
            reviews.append((reviewer, Date, Description, score1))
            
    return reviews

if __name__== "__main__":
    mpg_plot()
    
    movie_id='finding_dory'    
    reviews=getReviews(movie_id) 
    
    
    
    print(reviews)


# In[ ]:




