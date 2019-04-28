# Import Libraries
import pandas as pd
import string
import re
import os
import string
from nltk.stem import *
stemmer = PorterStemmer()
from nltk import corpus
import nltk



# Functions

def get_labeled_data(mydb):
    '''Retreive a dataframe of the Label + Text limited for a given Label'''
    sql_command = '''SELECT 
                           DL_2019_PROJ_A_LABELS.LABEL, 
                           DL_2019_PROJ_A_TEXT.TEXT

                     FROM GSU.DL_2019_PROJ_A_TEXT
                   
                     JOIN DL_2019_PROJ_A_LABELS ON
                          DL_2019_PROJ_A_TEXT.ID = DL_2019_PROJ_A_LABELS.ID    
                   '''
    df = pd.read_sql(sql_command, mydb)
    return df


def clean_and_tokenize_text(Text_file):
    '''
    Input      = Text File
    Operations = Tokenize, lowercase, strip punctuation/stopwords/nonAlpha
    Return     = Object = Set; Set = cleaned, isalpha only tokens
    '''

    # Strip Lists
    Punct_list = set((punct for punct in string.punctuation))
    Stopwords = nltk.corpus.stopwords.words('english')
    #Set_names = get_set_human_names()

    # Tokenize Text
    Text_tokenized = nltk.word_tokenize(Text_file)
    # Convert tokens to lowercase
    Text_lowercase = (token.lower() for token in Text_tokenized)
    # Strip Punctuation
    Text_tok_stripPunct = filter(lambda x: (x not in Punct_list), Text_lowercase)
    # Strip Stopwords
    Text_strip_stopWords = filter(lambda x: (x not in Stopwords), Text_tok_stripPunct)
    # Strip Non-Alpha
    Text_strip_nonAlpha = filter(lambda x: x.isalpha(), Text_strip_stopWords)
    # Strip 2 letter words
    Text_strip_2_letter_words = filter(lambda x: len(x)>2, Text_strip_nonAlpha)
    # Strip names
    #Text_strip_names = filter(lambda x: x not in Set_names, Text_strip_2_letter_words)
    # Take Stem of Words
    Text_stem = [stemmer.stem(x) for x in Text_strip_2_letter_words]
    # Convert Back to List
    Text_list = list(Text_stem)
    # Convert Back to String
    Text_str = ' '.join(Text_list)   
    return Text_str




