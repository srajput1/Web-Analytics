
# coding: utf-8

# In[272]:


#Sourabh Rajput
#BIA 660 C
#CWID 10431188


# Structure of your solution to Assignment 1 

import numpy as np
import csv

 
def count_token(text):  #Function Count_token, text is a parameter passed from main
    
    
      
    t = text.split(' ') #splits the string into a list of tokens by space
    tp = []
    token_count = dict()
    for i in t:
        i.strip()       #strips all leading and trailing space of each token
        i = i.replace('\n','') 
        if len(i) > 1: #removes a token if it contain no more than 1 character
            tp.append(i.lower()) #converts all tokens into lower case
   
    for i in tp:    #counts every remaining token
        if i in token_count:
            token_count[i] += 1
        else:
            token_count[i] = 1

     
     
    return token_count        #returns the dictionary

class Text_Analyzer(object):  #class Text_Analyzer
    
    def __init__(self, input_file, output_file): # input and out put files initialize with constructor
        self.input_file=input_file 
        self.output_file=output_file
        
    def analyze(self):    #function Analyze
        ReadFile = open(self.input_file,"r") 
        res=ReadFile.readlines(); #reads all lines from input_file
        str1 = ' '.join(res) #concatenate them into a string
        ReadFile.close()
        
        CallFunction={}
        CallFunction=count_token(str1) #calls the function "count_token"
         #saves the dictionary into output_file with each key-value pair as a line delimited by comma
        rows = list(CallFunction.items())
        
        with open("foo.csv", "w") as f:  
            writer=csv.writer(f, delimiter=",")
            writer.writerows(rows)
        # add your code

# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":  
    
    # Test Question 1
    text='''Hello world!
        This is a hello world example !'''   
    print(count_token(text))
    
    # # The output of your text should be: 
    # {'this': 1, 'is': 1, 'example': 1, 'world!': 1, 'world': 1, 'hello': 2}
    
# Test Question 2
    analyzer=Text_Analyzer("foo.txt", "foo.csv")
    vocabulary=analyzer.analyze()
    

