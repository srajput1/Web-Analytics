
# coding: utf-8

# In[236]:


#Sourabh Rajput
#BIA 660 C
#CWID 10431188


import csv
import nltk
import re 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def tokenize(text):
    text=text.lower() #converts to lower case
    t1 = nltk.word_tokenize(text) #tokenizes the lowercased string into tokens
    tokens=[] 
    tokens1=[]         
     
    s=[tokens for tokens in t1 if re.search(r'^[^-_0-9].*[^_-]$', tokens)] #token only contains letters (i.e. a-z or A-Z), "-" , or "_" . A token cannot starts or ends with "-" or "_" . 
    #print(s)
    for t in s:
        #t.strip()
        if len(t)>1:
            tokens1.append(t) #only those tokens are added which has more than one characters
    
    #print (tokens1)
    vocabulary= set(tokens1) 
    #print (vocabulary)
    stop_words = stopwords.words('english')
    
    #print (stop_words)
   
    for token in vocabulary:  #removes stop words and added to tokens
          if token not in stop_words:
                tokens.append(token)  
           
    
     
    return tokens # returns the resulting listing token as output


def sentiment_analysis(text, positive_words, negative_words):
    
    sentiment=None
    
    tokens = nltk.word_tokenize(text) #tokenizes the lowercased string into tokens    
    positive_tokens=[]
    negative_tokens=[]
    #negations=['not', 'n\'t',' no', 'cannot', 'neither', 'nor', 'too']
    
    positive_list=[token for token in tokens if token in positive_words] #counts positive words in each list
    #print(len(positive_list))
    
    negative_list=[token for token in tokens if token in negative_words] #counts negative words in each list
    #print(len(negative_list))
    
    for idx, token in enumerate(positive_list):
        if token in positive_words:
            if idx>0:
                if tokens[idx-1] not in negative_words: #a positive word not preceded by a negation word
                    positive_tokens.append(token)
                if tokens[idx-1]  in negative_words: #a negative word preceded by a negation word
                    negative_tokens.append(token)
                
            else:
                positive_tokens.append(token)
    #print(len(positive_tokens))
                
    for idx, token in enumerate(negative_list):
        if token in negative_words:
            if idx>0:
                if tokens[idx-1] not in negative_words: #a negative word not preceded by a negation word
                    negative_tokens.append(token)
                if tokens[idx-1]  in negative_words:# a positive word preceded by a negation word
                    positive_tokens.append(token)
            else:
                negative_tokens.append(token)
                
    #print(len(negative_tokens))

    if len(positive_tokens)>len(negative_tokens): # compares count of positive and negative word count
        sentiment = 2 # When positive word count is more then it will print statement 2
    else:
        sentiment = 1 #When negative word count is more then it will print statement 1
    
    return sentiment #returns the sentiment

def performance_evaluate(input_file, positive_words, negative_words):
    
    accuracy=None
    count = 0 #initialize counter with zero
    
    with open(input_file,'r') as f: #takes an input file ("amazon_review_300.csv"), a list of positive words, and a list of negative words as inputs. 
        data = [line for line in csv.reader(f)]  #The input file has a list of reviews in theformat of (label, title, review). Use label (either '2' or '1') and review columns (i.e. columns 1 and 3 only) here.
        
    #print(data)
    
    new=[]
    mew=[]

    for review,title,lable in data:# makes rewiew list 
        new.append(review)
        mew.append(lable)

    count_data = zip(new,mew)
    demo_review = list(count_data)
    for label, review in demo_review: #makes labe list 
        result = str(sentiment_analysis(review,positive_words, negative_words))
    
        if result == label: #counts
            count+=1  

    accuracy = count/len(demo_review) #calculating result in accuracy
    return accuracy #returns the accuracy as the number of correct sentiment predictions/total reviews



if __name__ == "__main__":  
    
    text="this is a  breath-taking  ambitious movie; test  text: abc_dcd abc_ dvr89w, abc-dcd -abc"

    tokens=tokenize(text)
    print("tokens:")
    print(tokens)
    
    with open("positive-words.txt",'r') as f:
        positive_words=[line.strip() for line in f]
        
    with open("negative-words.txt",'r') as f:
        negative_words=[line.strip() for line in f]
        #print(negative_words)
    print("\nsentiment")
    sentiment=sentiment_analysis(text, positive_words, negative_words)
    print(sentiment)
    
   
    accuracy=performance_evaluate("amazon_review_300.csv", positive_words, negative_words)
    
    print("\naccuracy")
    print(accuracy)

