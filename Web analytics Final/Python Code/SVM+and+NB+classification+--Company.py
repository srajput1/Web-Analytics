
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer

import csv

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_validate

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_validate

from sklearn import svm

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV


from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import classification_report


# In[2]:


#Selecting Each company and running one model for all
list_to_use1 = ['Microsoft','Netflix','amazon','Boeing','Google','Facebook']
for company in list_to_use1:
    with open("C:/Users/rajpu/Desktop/Web analytics Final/Data/"+company.lower()+".csv", "r") as f:
        data = [tuple(line) for line in csv.reader(f)]
    
    Lable,Date,News,Company,Sentiment,Open,High,Low,Close,Volume=zip(*data)

    date=list(Date)
    news=list(News)
    sent=list(Sentiment)
    lab=list(Lable)

    tfidf_vect = TfidfVectorizer() 
    tfidf_vect = TfidfVectorizer(stop_words="english") 

    dtm= tfidf_vect.fit_transform(news)
    print(company)
    print("type of dtm:", type(dtm))
    print("size of tfidf matrix:", dtm.shape)
    
    #print("type of vocabulary:", type(tfidf_vect.vocabulary_))
    #print("index of word 'city' in vocabulary:", \
    #      tfidf_vect.vocabulary_['city'])
    voc_lookup={tfidf_vect.vocabulary_[word]:word                 for word in tfidf_vect.vocabulary_}
    print("total number of words:", len(voc_lookup ))

    #print("\nOriginal text: \n"+news[1])

    #print("\ntfidf weights: \n")
    doc0=dtm[0].toarray()[0]
    print(doc0.shape)


    top_words=(doc0.argsort())[::-1][0:20]
    print([(voc_lookup[i], doc0[i]) for i in top_words])
    X_train, X_test, y_train, y_test = train_test_split(                dtm, sent, test_size=0.3, random_state=0)
    # train a multinomial naive Bayes model using the testing data
    clf = MultinomialNB().fit(X_train, y_train)

    predicted=clf.predict(X_test)

    labels=sorted(list(set(sent)))

    precision, recall, fscore, support=         precision_recall_fscore_support(         y_test, predicted, labels=labels)
    
#print(precision)
#print(recall)
#print(fscore)
#print(support)

    print(classification_report(y_test, predicted, target_names=labels))
    
    metrics = ['precision_macro', 'recall_macro', "f1_macro"]

    print('SVM Result')
    clf = svm.LinearSVC().fit(X_train, y_train)
    predicted=clf.predict(X_test)

    #labels=sorted(list(set(sent)))

    precision, recall, fscore, support=     precision_recall_fscore_support(     y_test, predicted, labels=labels)
    
#print(precision)
#print(recall)
#print(fscore)
#print(support)

    print(classification_report(y_test, predicted, target_names=labels))
    

