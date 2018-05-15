
# coding: utf-8

# In[1]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

from nltk.cluster import KMeansClusterer, cosine_distance

import json
from numpy.random import shuffle

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


from sklearn.decomposition import LatentDirichletAllocation

from scipy.cluster.hierarchy import cut_tree

from sklearn import metrics
import numpy as np

import csv

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn.metrics.pairwise import cosine_similarity


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.decomposition import LatentDirichletAllocation

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import math


# In[20]:



list_to_use1 = ['Microsoft_neg','Microsoft_pos','Netflix_neg','Netflix_pos','amazon_neg',                'amazon_pos','Boeing_neg','Boeing_pos','Google_neg','Google_pos','Facebook_neg','Facebook_pos']
for company in list_to_use1:
    with open("C:/Users/rajpu/Desktop/Web analytics Final/Data/"+company+".csv", "r") as f:
        reader=csv.reader(f, delimiter=',') 
        rows=[(row[1][2],row[1][4])           for row in enumerate(reader)]
    shuffle(rows)
#rows
    text,label=zip(*rows)
    text=list(text)
    label=list(label)
   
    # LDA can only use raw term counts for LDA 
    tf_vectorizer = CountVectorizer(max_df=0.90,                     min_df=5, stop_words='english')
    tf = tf_vectorizer.fit_transform(text)
    
    # each feature is a word (bag of words)
    # get_feature_names() gives all words
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    #print(tf_feature_names[0:10])
    #print(tf.shape)
    
    
    # split dataset into train (90%) and test sets (10%)
    # the test sets will be used to evaluate proplexity of topic modeling
    X_train, X_test = train_test_split(                    tf, test_size=0.1, random_state=0)
    
    num_topics = 1

    # Run LDA. For details, check
    # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation.perplexity

    # max_iter control the number of iterations 
    # evaluate_every determines how often the perplexity is calculated
    # n_jobs is the number of parallel threads
    lda = LatentDirichletAllocation(n_components=num_topics,                                         max_iter=10,verbose=1,
                                        evaluate_every=1, n_jobs=1,
                                        random_state=0).fit(X_train)
    num_top_words=20
    print(company)

    # lda.components_ returns a KxN matrix
    # for word distribution in each topic.
    # Each row consists of 
    # probability (counts) of each word in the feature space

    for topic_idx, topic in enumerate(lda.components_):
        print ("Topic %d:" % (topic_idx))
        # print out top 20 words per topic 
        words=[(tf_feature_names[i],topic[i]) for i in topic.argsort()[::-1][0:num_top_words]]
        print(words)
        print("\n")
        num_top_words=50
        #f, axarr = plt.subplots(2, 2, figsize=(8, 8));

        #for topic_idx, topic in enumerate(lda.components_):
    # create a dataframe with two columns (word, weight) for each topic
    
    # create a word:count dictionary
           # f={tf_feature_names[i]:topic[i] for i in topic.argsort()[::-1][0:num_top_words]}
    
    # generate wordcloud in subplots
    for topic_idx, topic in enumerate(lda.components_):
        f = {tf_feature_names[i]:topic[i] for i in
                topic.argsort()[::-1][0:num_top_words]}
        wordcloud = WordCloud(width=480,height=450,margin=0,background_color='black')
        wordcloud.generate_from_frequencies(frequencies=f)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

