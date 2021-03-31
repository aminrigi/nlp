"""
This file contains a few hany function for text cleaning and plotting.
"""


custom_stopwords = ['dodo', 'service', 'nt']

import spacy
nlp = spacy.load('en_core_web_sm')
def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)

import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english') + custom_stopwords
nltk.download('punkt')


def  clean_text(df, text_field, new_text_field_name):
    """
    This function cleans a text column in a dataframe. 
    Note: this function can get much much faster if lemmatizing is ignored
    
    Args:
    df: input dataframe
    text_field: name of the col that corresponds to text
    new_text_field_name : name of the cleaned text col
    
    Usage:
    df = clean_text(df, 'comments', 'comments_cleaned') 
    """
    df[new_text_field_name] = df[text_field].str.lower()
    
    #remove numbers and urls
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    #removing stopwords
    df[new_text_field_name] = df[new_text_field_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    #lemmatize
    df[new_text_field_name] = df[new_text_field_name].apply(lemmatizer)

    return df




from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
import pandas as pd

def plot_top_n_words(corpus, n):
    """
    This function plots the most frequent words for a string column of a df
    
    
    Args:
    corpus: text col of a dataframe
    n: numbre of top words to be shown
    """
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    count_df = pd.DataFrame(words_freq[:n], columns = ['unigram' , 'count']).sort_values('count')
    fig = go.Figure(go.Bar( y=count_df['unigram'], x=count_df['count'],  orientation='h'))
    fig.show()
    
    
def plot_top_n_bigram(corpus, n):
    """
    This function plots the most frequent bigrams for a string column of a df
    
    
    Args:
    corpus: text col of a dataframe
    n: numbre of top bigrams to be shown
    """
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    count_df = pd.DataFrame(words_freq[:n], columns = ['bigram' , 'count']).sort_values('count')
    fig = go.Figure(go.Bar( y=count_df['bigram'], x=count_df['count'],  orientation='h'))
    fig.show() 
