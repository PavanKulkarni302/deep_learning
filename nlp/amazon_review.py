# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 08:40:29 2024

@author: pavan
"""

import pandas as pd
reviews = pd.read_csv("sentiment_labelled_sentences/sentiment labelled sentences/amazon_cells_labelled.txt",
                      sep='\t', names= ['reviews','rating'])

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

ps = PorterStemmer()
lm = WordNetLemmatizer()
cleaned_reviews = []

for i in range(0,len(reviews)):
    sent = re.sub('[^a-zA-z]', ' ', reviews['reviews'][i])
    sent = sent.lower()
    words = nltk.word_tokenize(sent)
    word = [lm.lemmatize(word) for word in words if word not in stopwords.words('english')]
    word = ' '.join(word)
    cleaned_reviews.append(word)
    

#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=1200)
X = tfidf.fit_transform(cleaned_reviews).toarray()
y = reviews['rating'] 

#split the data for test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state=42)


#build model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
model = nb.fit(X_train, y_train)

#prediction on test data
y_pred = model.predict(X_test)


#model validation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)











