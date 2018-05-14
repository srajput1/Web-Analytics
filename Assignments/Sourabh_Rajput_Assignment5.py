
# coding: utf-8

# In[121]:


#Sourabh Rajput
#BIA 660 C
#CWID 10431188

import nltk
import csv
import os,sys
import nltk, re, string
from sklearn.preprocessing import normalize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from scipy.spatial import distance
import numpy as np  
import pandas as pd


def top_collocation(tokens, K):
    result=[]
    
    a= nltk.pos_tag(tokens)#POS tag each token
    #print(a)
    
    bigrams=list(nltk.bigrams(a)) #create bigrams
    #print(bigrams)
    #get frequency of each bigram (you can use nltk.FreqDist)
    word_dist=nltk.FreqDist(bigrams) 
    #print(word_dist)
    phrases=[ (x[0],y[0]) for (x,y) in bigrams          if x[1].startswith('JJ')and y[1].startswith('NN')
         or x[1].startswith('NN')and y[1].startswith('NN')]
    
    result=nltk.FreqDist(phrases)
    return result.most_common(K)
    
    return result
def get_doc_tokens(doc,normalize):
    
    stop_words = stopwords.words('english') 
    tokens1=[token.strip()             for token in nltk.word_tokenize(doc.lower())             if token.strip() not in stop_words and               token.strip() not in string.punctuation]
    if normalize== None:
        tokens=tokens1
    
    elif normalize == 'stem':
        porter_stemmer = PorterStemmer()
        tokens = [porter_stemmer.stem(str(token)) for token in tokens1]
        
    
    # you can add bigrams, collocations, stemming, 
    # or lemmatization here
    
    token_count={token:tokens.count(token) for token in set(tokens)}
    return token_count

def tfidf(docs,normalize):
    #docs_tokens={idx:get_doc_tokens(doc) \
     #        for idx,doc in enumerate(docs)}
    #print(docs_tokens)
    docs2=[]
    if normalize == None:
        docs1={idx:get_doc_tokens(doc, None) for idx,doc in enumerate(docs)}
    elif normalize == 'stem':
        docs1={idx:get_doc_tokens(doc, 'stem') for idx,doc in enumerate(docs)}
 
    dtm=pd.DataFrame.from_dict(docs1, orient="index" )
    dtm=dtm.fillna(0)
    #print(dtm)
     
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf.T, doc_len).T
    #print(tf)
        
    df=np.where(tf>0,1,0)
    #print(df)
       
    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1    
    smoothed_tf_idf=tf*smoothed_idf
    
    
    
    return smoothed_tf_idf
    #print(docs)
    
    
    
    

if __name__ == "__main__":  
    
    # test collocation
    text=nltk.corpus.reuters.raw('test/14826')
    tokens=nltk.word_tokenize(text.lower())
    print(top_collocation(tokens, 10))
    
    docs=[]
    with open("amazon_review_300.csv","r") as f:
        reader=csv.reader(f)
    
        for line in reader:
            docs.append(line[2])
            #docs = ['Hate organizing this big bull hell']
    #tfidf=tfidf(docs,normalize) 
    
    # Find similar documents -- No STEMMING
    print("Result without stemming")
    result1 = tfidf(docs,None)
    print(result1)
    no_similarity=1-distance.squareform (distance.pdist(result1, 'cosine'))
    print(no_similarity)
    nonstem = np.argsort(no_similarity)[:,::-1][0,0:6]
    print(nonstem)
    for idx, doc in enumerate(docs):
        if idx in nonstem:
            print(idx,doc)
    # Find similar documents -- STEMMING  
    print("Result with Stemming")
    result = tfidf(docs,'stem')
    print(result)
    similarity_stem=1-distance.squareform (distance.pdist(result, 'cosine'))
    print(similarity_stem)    
    np.argsort(similarity_stem)
    top5_stem = np.argsort(similarity_stem)[:,::-1][0,0:6]  
    print(top5_stem)
    
    for idx, doc in enumerate(docs):
        if idx in top5_stem:
            print(idx,doc)
            
    print("Analysis")
    print("In the above case the Result without Stemming is good  because it finds perfect similarities by comparing root words with each others \n It finds 3 positive words and 2 negative words where in other result with stem finds 3 negative and 2 positive words.")
    
    
    
    #print(docs)
    # Find similar documents -- No STEMMING
    
    # Find similar documents -- STEMMING  

