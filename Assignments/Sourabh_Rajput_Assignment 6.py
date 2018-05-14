#Sourabh Rajput
#BIA 660 C
#10431188

# coding: utf-8

# In[10]:


import csv
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt


# In[83]:


with open('amazon_review_300.csv', 'r') as f:
    data = [tuple(line) for line in csv.reader(f)]

    target,nr,text=zip(*data)
    
    target=list(target)
    text=list(text)

    tfidf_vect = TfidfVectorizer(stop_words = "english") 
    dtm = tfidf_vect.fit_transform(text)
    metrics = ['precision_macro', 'recall_macro', "f1_macro"]
    clf = MultinomialNB(alpha=0.8)
    cv = cross_validate(clf, dtm, target, scoring=metrics, cv=6)
    
    print("Naive Bayes results: \n")
    print("Test data set average precision:")
    print(cv['test_precision_macro'])
    print("\nTest data set average recall:")
    print(cv['test_recall_macro'])
    print("\nTest data set average fscore:")
    print(cv['test_f1_macro'])


# In[104]:


with open('amazon_review_300.csv', 'r') as f:
    data = [tuple(line) for line in csv.reader(f)]
        
    stop_words = stopwords.words('english')

    target,nr,text=zip(*data)
    
    target=list(target)
    text=list(text)
    
    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])

    parameters = {'tfidf__min_df':[1, 2, 3, 5],
                  'tfidf__stop_words':[None, "english"],
                  'clf__alpha': [0.5, 1.0, 1.5, 2.0],}

    metric =  "f1_macro"

    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=6)
    gs_clf = gs_clf.fit(text, target)
    print("\n")
    
    for param_name in gs_clf.best_params_:
        print(param_name,": ",gs_clf.best_params_[param_name])

    print("Best f1 score:", gs_clf.best_score_)
    
    metrics = ['precision_macro', 'recall_macro', "f1_macro"]
    
    tfidf_vect = TfidfVectorizer(stop_words = "english", min_df = 5) 
    gs_dtm = tfidf_vect.fit_transform(text)
    gs_clf = MultinomialNB(alpha = 0.5)
    gs_cv = cross_validate(gs_clf, gs_dtm, target, scoring=metrics, cv=6)
    
    print("\nGrid search results: \n")
    print("Test data set average precision:")
    print(gs_cv['test_precision_macro'])
    print("\nTest data set average recall:")
    print(gs_cv['test_recall_macro'])
    print("\nTest data set average fscore:")
    print(gs_cv['test_f1_macro'])


# In[126]:


list1=[]
with open('amazon_review_large.csv','r') as f:
    reader=csv.reader(f)
    list1=[(line[0], line[1]) for line in reader]

y, x= zip(*list1)
Y=list(y)
X=list(x)

metrics = ['precision_macro', 'recall_macro', "f1_macro"]

r=[]
r1=[]
tfidf_vect = TfidfVectorizer(stop_words="english") 
clf = svm.LinearSVC()
clf1 = MultinomialNB(alpha=0.8)
s=400

while s<20000:
    dtm= tfidf_vect.fit_transform(X[0:s])
    cv = cross_validate(clf, dtm, y[0:s], scoring=metrics, cv=10)
    r.append((s, np.mean(np.array(cv['test_precision_macro'])),                   np.mean(np.array(cv['test_recall_macro'])),                   np.mean(np.array(cv['test_f1_macro']))))
    s+=400 



s1=400
while s1<20000:
    dtm= tfidf_vect.fit_transform(X[0:s1])
    cv = cross_validate(clf1, dtm, y[0:s1], scoring=metrics, cv=10)
    r1.append((s1, np.mean(np.array(cv['test_precision_macro'])),                   np.mean(np.array(cv['test_recall_macro'])),                   np.mean(np.array(cv['test_f1_macro']))))
    s1+=400 
    
r=np.array(r)
r1=np.array(r1)


plt.plot(r[:,0], r[:,3], '-', label='f1')
plt.plot(r1[:,0], r1[:,3], '-', label='f1 multi')


plt.title('Impact of sample size on classication performance')
plt.ylabel('performance')
plt.xlabel('sample size')
plt.legend()
plt.show()

