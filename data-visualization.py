# %matplotlib inline
#imports 
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidataVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup



data = pd.read_csv('/Users/jeff/Documents/ML2/CarTalk_Topic_Attributes_5.csv')
# print(data.shape) #prints row and column quantity in that order
data = data[[ 'Unnamed: 0', 'Topic Title', 'Category', 'Tags', 'Commenters',
        'Leading Comment', 'Other Comments', 'Likes', 'Views']]

# data['Likes']=data.apply(lambda x: x['Likes']+3,axis=1) 

data.drop(columns = ['Unnamed: 0', 'Tags'], inplace = True)
table = data.head()
print(table)



# print(data.shape) #updated quantity
# data['Leading Comment'][2]
# print(data.isnull().sum())
# data = data.groupby('Likes').filter(lambda x: 'Nice' if len(x)>10).reset_index(drop=True)
# print(data)
# print(len(data['Likes'].unique()))
# data.info()
# print(data.columns)
# data[['Unnamed: 0', 'Topic Title', 'Category', 'Tags', 'Author', 'Commenters',
#        'Leading Comment', 'Other Comments', 'Likes', 'Views']]
# data.drop(columns = ['Unnamed: 0', 'Topic Title', 'Category', 'Tags', 'Author', 'Commenters',
#        'Leading Comment', 'Other Comments', 'Likes', 'Views'], inplace=True)
# data['Topic Title'][1]
# all_categ = data['Category'].unique()
# print(all_categ)
# print(len(all_categ))