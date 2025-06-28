# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 21:01:36 2024

@author: pavan
"""

import pandas as pd
messages = pd.read_csv("sms_spam_collection/SMSSpamCollection",sep= '\t',
                 names= ['label','message'])

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
ps = PorterStemmer()
lm = WordNetLemmatizer()
corpus = []

for i in range(len(messages)):
    sent = re.sub('[^a-zA-z]', ' ',messages['message'][i])
    sent = sent.lower()
    words = nltk.word_tokenize(sent)
    word = [lm.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    word = ' '.join(word)
    corpus.append(word)
    
    
#creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000)
X = cv.fit_transform(corpus).toarray()


y = pd.get_dummies(messages['label']).astype(int)
y = y.iloc[:,1]



#spliting X and y data into training data and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#building model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
spam_detect_model = model.fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

#confussion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)


from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)





