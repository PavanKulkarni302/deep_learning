# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:16:32 2024

@author: pavan
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""

#data preprocessing
text = re.sub(r'\[[0-9]*\]',' ',paragraph) #remove numerical references enclosed in square brackets from the text.
text = re.sub(r'\s+',' ',text) #replaces multiple consecutive whitespace characters (spaces, tabs, newlines) with a single space.
text = text.lower() 
text = re.sub(r'\d',' ',text) #replaces all digits in the text with a space.
text = re.sub(r'\s+',' ',text) #after removing digits - replaces multiple consecutive whitespace characters (spaces, tabs, newlines) with a single space.

sentences = nltk.sent_tokenize(text)
lm = WordNetLemmatizer()
#removing stop words and stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    sentences[i] = [lm.lemmatize(word) for word in words if word not in stopwords.words('english')]

#training word2vec model
from gensim.models import Word2Vec
model = Word2Vec(sentences, min_count=1)


#see vocabolary in the model
words_w2v = model.wv.index_to_key


#finding word vectors
word_vec = model.wv['succeeded']

#finding similar wors
words_similar = model.wv.most_similar('succeeded')


