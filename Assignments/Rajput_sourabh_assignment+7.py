
# coding: utf-8

# In[60]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

from nltk.cluster import KMeansClusterer, cosine_distance

import json
from numpy.random import shuffle

import pandas as pd

from sklearn import metrics
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.decomposition import LatentDirichletAllocation

from scipy.cluster.hierarchy import cut_tree


# In[143]:


data=json.load(open('ydata_3group.json','r'))


shuffle(data)

text,target,nr=zip(*data)
text=list(text)
target=list(target)
#label=list(label)


data=[item for item in data if item[1] in       ['T1','T2', 'T3']]
tfidf_vect = TfidfVectorizer(stop_words="english",                             min_df=5) 

# generate tfidf matrix
dtm= tfidf_vect.fit_transform(text)
#print (dtm.shape)

num_clusters=3
clusterer = KMeansClusterer(num_clusters,                             cosine_distance, repeats=10)
clusters = clusterer.cluster(dtm.toarray(),                             assign_clusters=True)

#print(clusters[0:5])

df=pd.DataFrame(list(zip(target, clusters)),                 columns=['actual_class','cluster'])
df.head()
pd.crosstab( index=df.cluster, columns=df.actual_class)


cluster_dict={0:'T3', 1:"T1",              2:'T2'}

# Assign true class to cluster
predicted_target=[cluster_dict[i] for i in clusters]

print(metrics.classification_report      (target, predicted_target))
centroids=np.array(clusterer.means())

sorted_centroids = centroids.argsort()[:, ::-1] 

voc_lookup= tfidf_vect.get_feature_names()

for i in range(num_clusters):
    
    # get words with top 5 tf-idf weight in the centroid
    top_words=[voc_lookup[word_index]                for word_index in sorted_centroids[i, :5]]
    print("Cluster %d: %s " % (i, "; ".join(top_words)))







# In[148]:



tf_vectorizer = CountVectorizer(max_df=0.90,                 min_df=50, stop_words='english')
tf = tf_vectorizer.fit_transform(text)
tf_feature_names = tf_vectorizer.get_feature_names()
print(tf_feature_names[0:10])
print(tf.shape)
num_topics = 3
lda = LatentDirichletAllocation(n_components=num_topics,                                 max_iter=10,verbose=1,
                                evaluate_every=1, n_jobs=1,
                                random_state=0).fit(tf)
num_top_words=5
for topic_idx, topic in enumerate(lda.components_):
    print ("Topic %d:" % (topic_idx))
    # print out top 20 words per topic 
    words=[(tf_feature_names[i],topic[i]) for i in topic.argsort()[::-1][0:num_top_words]]
    print(words)
    print("\n")
topic_assign=lda.transform(tf)
print(topic_assign[0:5])
topics=np.copy(topic_assign)
xyz=np.argsort(topics)
abc1=xyz[:,::-1]
abc1=[(row[1][0])           for row in enumerate(abc1)]
df=pd.DataFrame(list(zip(target, abc1)),                 columns=['actual_class','Topic'])
df.head()
pd.crosstab( index=df.Topic, columns=df.actual_class)
cluster_dict={0:'T3', 1:"T1",              2:'T2'}
predicted_target=[cluster_dict[i] for i in abc1]
print(metrics.classification_report      (target, predicted_target))

