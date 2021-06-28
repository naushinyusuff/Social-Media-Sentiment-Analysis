
#importing the necessary libraries
import nltk
import string
import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import sklearn
import joblib

# A dictionary containing some of the common emoticons
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

def replace_emoticon(text):
    """
    This function replaces emoticons with their equivalent emotion
    """
    for emoji in emojis.keys():
        text = text.replace(emoji, " EMOJI" + emojis[emoji] + " ") 
    return text
    
punc = string.punctuation
punc = punc.replace("&","")

def remove_punctuation(text):
    """
    This function removes all the punctuations included in the string punc with a blank space
    """
    s =[]
    for char in text:
        if char in punc:
            s.append(" ")
        else:
            s.append(char)           
    return "".join(s)

def remove_ampersands(text):
    """
    When the dataset is scrapped from the some unecessary ampersanded words can arise which are removed by this function
    """
    words = text.split();
    n = len(words)
    for i in range(0,n):
        if "&" in words[i]:
            temp = words[i].split("&")
            if len(temp) == 1:
                words[i]  =""
            else:
                words[i] = temp[0]
    return " ".join(words)    

def remove_numbers(text):
    """
    This function removes numbers
    """
    result = re.sub(r'\d+', '', text)
    return result

def util_func(text):
    n = len(text)
    flag  = 0;
    char = ""
    s = []
    for i in range(0,n):
        if char!=text[i]:
            flag = 1
            char = text[i]
            s.append(char)
            continue            
        if char==text[i] and flag==1:
            flag = 2
            s.append(char)
            continue            
        if flag==2 and char==text[i]:
            continue            
    return "".join(s)       


def remove_morethan2letters(text):
    """
    if more than 2 letters occur consecutively they are replaced by a 2 letters
    """
    words = text.split();
    n = len(words)
    for i in range(0,n):
        words[i] = util_func(words[i])
        
    return " ".join(words) 


#Stemming
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer = PorterStemmer()

def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    stems = ' '.join(stems)
    return stems    
    
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer = WordNetLemmatizer()

def lemmatize_word(text):
    """
    Lemmatizes the text
    """
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in tokens]
    lemmas = ' '.join(lemmas)
    return lemmas 

#Stopwords
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words.remove('no')
stop_words.remove('not')
stop_words.remove("nor")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def remove_stopwords(text):
    """
    Removes stopwords from the text
    """
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    filtered = ' '.join(filtered)
    return filtered


#the analyser function which includes all the aboove functions
def text_preprocessor(text):

    text = text.lower()
    text = replace_emoticon(text)    
    text = remove_punctuation(text) 
    text = remove_ampersands(text)
    text = remove_numbers(text)
    text = remove_morethan2letters(text)
    text = lemmatize_word(text)
    text = remove_stopwords(text)
    
    return text


model = joblib.load('Sentiment_Analysis_model')
st.title('Sentiment Analyzer For Tweets')
st.write("Type in a tweet and get to know the emotion it radiates!")
ip = st.text_input("Enter the tweet: ")
ip = text_preprocessor(ip)
op = model.predict([ip])
if op==0:
    emotion = "The tweet gives Negative vibes!"
if op==4: 
    emotion = "The tweet gives Positive vibes!"
if st.button("Predict"):
  st.title(emotion)
