
# coding: utf-8

# In[89]:


import json
import pandas as pd
import numpy as np
import csv
from gensim.models import word2vec
import logging
import gensim
import nltk, string
from sklearn.preprocessing import MultiLabelBinarizer
from numpy.random import shuffle
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Input, Flatten, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import gensim


# In[79]:


def cnn_model(FILTER_SIZES,               # filter sizes as a list
              MAX_NB_WORDS, \
              # total number of words
              MAX_DOC_LEN, \
              # max words in a doc
              EMBEDDING_DIM=200, \
              # word vector dimension
              NUM_FILTERS=64, \
              # number of filters for all size
              DROP_OUT=0.5, \
              # dropout rate
              NUM_OUTPUT_UNITS=1, \
              # number of output units
              NUM_DENSE_UNITS=100,\
              # number of units in dense layer
              PRETRAINED_WORD_VECTOR=None,\
              # Whether to use pretrained word vectors
              LAM=0.0):            
              # regularization coefficient
    
    main_input = Input(shape=(MAX_DOC_LEN,),                        dtype='int32', name='main_input')
    
    if PRETRAINED_WORD_VECTOR is not None:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1,                         output_dim=EMBEDDING_DIM,                         input_length=MAX_DOC_LEN,                         weights=[PRETRAINED_WORD_VECTOR],                        trainable=False,                        name='embedding')(main_input)
    else:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1,                         output_dim=EMBEDDING_DIM,                         input_length=MAX_DOC_LEN,                         name='embedding')(main_input)
    # add convolution-pooling-flat block
    conv_blocks = []
    for f in FILTER_SIZES:
        conv = Conv1D(filters=NUM_FILTERS, kernel_size=f,                       activation='relu', name='conv_'+str(f))(embed_1)
        conv = MaxPooling1D(MAX_DOC_LEN-f+1, name='max_'+str(f))(conv)
        conv = Flatten(name='flat_'+str(f))(conv)
        conv_blocks.append(conv)
    
    if len(conv_blocks)>1:
        z=Concatenate(name='concate')(conv_blocks)
    else:
        z=conv_blocks[0]
        
    drop=Dropout(rate=DROP_OUT, name='dropout')(z)

    dense = Dense(NUM_DENSE_UNITS, activation='relu',                    kernel_regularizer=l2(LAM),name='dense')(drop)
    preds = Dense(NUM_OUTPUT_UNITS, activation='sigmoid', name='output')(dense)
    model = Model(inputs=main_input, outputs=preds)
    
    model.compile(loss="binary_crossentropy",               optimizer="adam", metrics=["accuracy"]) 
    
    return model


# In[80]:


data=pd.read_csv('train.csv',encoding='Latin1', header=None)
trainDataReview=data[0]
trainDataSentiment=data[1]
del trainDataSentiment[0]
del trainDataReview[0]
#data
data1=pd.read_csv('test.csv',encoding='Latin1', header=None)
#data1
testdata1_Review=data1[0]
testdata1_Sentiment=data1[1]
del testdata1_Review[0]
del testdata1_Sentiment[0]


# In[81]:



# set the maximum number of words to be used
MAX_NB_WORDS=10000

# set sentence/document length
MAX_DOC_LEN=1000

# get a Keras tokenizer
# https://keras.io/preprocessing/text/
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(trainDataReview)

# convert each document to a list of word index as a sequence
sequences = tokenizer.texts_to_sequences(trainDataReview)

# pad all sequences into the same length 
# if a sentence is longer than maxlen, pad it in the right
# if a sentence is shorter than maxlen, truncate it in the right
padded_seq = pad_sequences(sequences,                                  maxlen=MAX_DOC_LEN,                                  padding='post',                                  truncating='post')

test_seq = tokenizer.texts_to_sequences(testdata1_Review)

# pad all sequences into the same length 
# if a sentence is longer than maxlen, pad it in the right
# if a sentence is shorter than maxlen, truncate it in the right
padded_test_seq = pad_sequences(test_seq,                                  maxlen=MAX_DOC_LEN,                                  padding='post',                                  truncating='post')

EMBEDDING_DIM=300
FILTER_SIZES=[2,3,4]

# set the number of output units
# as the number of classes
output_units_num=1
num_filters=64

# set the dense units
dense_units_num= num_filters*len(FILTER_SIZES)

BTACH_SIZE = 32
NUM_EPOCHES = 100

BEST_MODEL_FILEPATH='best_model'

# With well trained word vectors, sample size can be reduced
# Assume we only have 500 labeled data
# split dataset into train (70%) and test sets (20%)

# create the model with embedding matrix
model=cnn_model(FILTER_SIZES, MAX_NB_WORDS,                 MAX_DOC_LEN,                 EMBEDDING_DIM=300,                NUM_OUTPUT_UNITS=output_units_num,                 NUM_FILTERS=num_filters,                NUM_DENSE_UNITS=dense_units_num)

earlyStopping=EarlyStopping(monitor='val_loss', patience=1, verbose=2, mode='min')
checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_acc',                              verbose=2, save_best_only=True, mode='max')
    
training=model.fit(padded_seq, trainDataSentiment,           batch_size=BTACH_SIZE, epochs=NUM_EPOCHES,           callbacks=[earlyStopping, checkpoint],          validation_data=[padded_test_seq, testdata1_Sentiment], verbose=2)



# In[82]:


print(training.history)

model.load_weights("best_model")

# predict
pred=model.predict(padded_test_seq)
pred=np.where(pred>0.5,1,0)
print(pred[0:5])
# evaluate the model
scores = model.evaluate(padded_test_seq, testdata1_Sentiment, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

setimentTestNum=pd.to_numeric(testdata1_Sentiment)
setimentTrainNum=pd.to_numeric(trainDataSentiment)
print(classification_report(setimentTestNum, pred))


# In[83]:




wv_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 


# In[84]:


MAX_NB_WORDS=12000
EMBEDDING_DIM=300

# tokenizer.word_index provides the mapping 
# between a word and word index for all words
NUM_WORDS = min(MAX_NB_WORDS, len(tokenizer.word_index))

# "+1" is for padding symbol
embedding_matrix = np.zeros((NUM_WORDS+1, EMBEDDING_DIM))

ignored_words=[]
for word, i in tokenizer.word_index.items():
    # if word_index is above the max number of words, ignore it
    if i >= NUM_WORDS:
        continue
    if word in wv_model.wv:
        embedding_matrix[i]=wv_model.wv[word]
    else:
        ignored_words.append(word)
        


# In[86]:


# set the maximum number of words to be used
MAX_NB_WORDS=12000

# set sentence/document length
MAX_DOC_LEN=1000

# get a Keras tokenizer
# https://keras.io/preprocessing/text/
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(trainDataReview)

# convert each document to a list of word index as a sequence
sequences = tokenizer.texts_to_sequences(trainDataReview)

# pad all sequences into the same length 
# if a sentence is longer than maxlen, pad it in the right
# if a sentence is shorter than maxlen, truncate it in the right
padded_sequences = pad_sequences(sequences,                                  maxlen=MAX_DOC_LEN,                                  padding='post',                                  truncating='post')

test_sequences = tokenizer.texts_to_sequences(testdata1_Review)

# pad all sequences into the same length 
# if a sentence is longer than maxlen, pad it in the right
# if a sentence is shorter than maxlen, truncate it in the right
padded_test_sequences = pad_sequences(test_sequences,                                  maxlen=MAX_DOC_LEN,                                  padding='post',                                  truncating='post')




# In[87]:



EMBEDDING_DIM=300
FILTER_SIZES=[2,3,4]

# set the number of output units
# as the number of classes
output_units_num=1
num_filters=64

# set the dense units
dense_units_num= num_filters*len(FILTER_SIZES)

BTACH_SIZE = 32
NUM_EPOCHES = 100

# With well trained word vectors, sample size can be reduced
# Assume we only have 500 labeled data
# split dataset into train (70%) and test sets (20%)


# create the model with embedding matrix
model=cnn_model(FILTER_SIZES, MAX_NB_WORDS,                 MAX_DOC_LEN,                 EMBEDDING_DIM=300,                NUM_OUTPUT_UNITS=output_units_num,                 NUM_FILTERS=num_filters,                NUM_DENSE_UNITS=dense_units_num,                PRETRAINED_WORD_VECTOR=embedding_matrix)

earlyStopping=EarlyStopping(monitor='val_acc', patience=3, verbose=2, mode='max')
checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_acc',                              verbose=2, save_best_only=True, mode='max')
    
training=model.fit(padded_sequences, trainDataSentiment,           batch_size=BTACH_SIZE, epochs=NUM_EPOCHES,           callbacks=[earlyStopping, checkpoint],          validation_data=[padded_test_sequences, testdata1_Sentiment], verbose=2)


# In[88]:


print(training.history)

model.load_weights("best_model")

# predict
pred=model.predict(padded_test_seq)
pred=np.where(pred>0.5,1,0)
print(pred[0:5])
# evaluate the model
scores = model.evaluate(padded_test_seq, testdata1_Sentiment, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

setimentTestNum=pd.to_numeric(testdata1_Sentiment)
setimentTrainNum=pd.to_numeric(trainDataSentiment)
print(classification_report(setimentTestNum, pred))

