#!/usr/bin/env python
# coding: utf-8

# David Signoretti
# 2019-07-16
# DeepLearning 3525

# Project Description

# I am using LSTM to predict the writings of Plato. But first, create a word2vec to generate a two-dimension vector image of the words. 
# 
# Generate the words' base on a prediction from a sample sentence.

# Import required Modules

import pandas as pd
from gensim.summarization.textcleaner import split_sentences
from gensim.models.word2vec import Word2Vec
from keras.callbacks import LambdaCallback, History, TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import random
import io
import sys
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import time


# Import text file of the the book Plato Republic

with io.open('Rep.txt', encoding='utf-8') as _f:
    _input_text = _f.read().lower()

_input_text = re.sub("\n", " ", _input_text)
_input_text = re.sub("\+", "", _input_text)
_input_text = re.sub("-", "", _input_text)
_input_text = re.sub("=", "", _input_text)
_input_text = re.sub("\[", "", _input_text)
_input_text = re.sub("\]", "", _input_text)
_input_text = re.sub("\(", "", _input_text)
_input_text = re.sub("\)", "", _input_text)

_length_of_input_text = len(_input_text)

print('_length_of_input_text = ',_length_of_input_text)

#create a sorted list of all the chaarcters from _input_text 
_sorted_characters = sorted(list(set(_input_text)))
_sorted_characters_length = len(_sorted_characters)
#create a list of words fron the _input_text
_list_of_tokenized_sentences = [_x.split() for _x in list(split_sentences(_input_text))]

print('_sorted_characters = ', _sorted_characters)
print('_sorted_characters_length = ', _sorted_characters_length)
print('_list_of_tokenized_sentences =', len(_list_of_tokenized_sentences))

#create a list unique words from the _input_text
_vocabulary_of_corpus = []
for _s in _list_of_tokenized_sentences:
    for _t in _s:
        if _t not in _vocabulary_of_corpus:
            _vocabulary_of_corpus.append(_t)

_vocabulary_size = len(_vocabulary_of_corpus)
print('Size of vocabulary = ', _vocabulary_size)

#create a dictionary charaters to a number to deconstruct words to numbers 
_words_to_index = {_w: _idx for (_idx, _w) in enumerate(_sorted_characters)}
#create a dictionary on numbers to character to create new words from perdictions
_index_to_words = {_idx: _w for (_idx, _w) in enumerate(_sorted_characters)}

print('_words_to_index = ', _words_to_index)


# Word2Vec Gensim


#create a word2vec 300 deminsion vector to display word relationships
_word_to_vec = Word2Vec(_list_of_tokenized_sentences, sg=1, seed=1, workers=3, size=300, min_count=3, window=7, sample=0.001)

_w = _word_to_vec[_word_to_vec.wv.vocab]

#convert a 300 dimensional array on to a 2 dimensions for human readable
#t-distributed Stochastic Neighbor Embedding
_tsne = TSNE(n_components=2, random_state=0)
_tsne_fit = _tsne.fit_transform(_w)

# Plot the word2vec in 2 dimensions


cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
#create a dataframe of the 2 dimensional array 
_df = pd.DataFrame(_tsne_fit, columns=['x', 'y'])
_df['token'] = _word_to_vec.wv.vocab.keys()
#plot
plt.figure(figsize=(15,8))
sns.scatterplot(_df.x, _df.y, palette=cmap)


# Look at word relationship

_word_to_vec.similar_by_word('plato'

_word_to_vec.similar_by_word('republic')


# Start of LSTM

#the maximum length of each sentence. There are 128 nodes in the LSTM layer and 64 is half 128
_max_length = 64
#number of steps is 4 because 128 and 64 are divisable by 4 
_number_of_steps = 4
#create empty arrays
_prepared_sentences = []
_prepared_char = []
#create an array of elements that are 64 charaters long, and each array is indexed by 4 charaters
for _i in range(0, _length_of_input_text - _max_length, _number_of_steps):
    _prepared_sentences.append(_input_text[_i: _i + _max_length])
    _prepared_char.append(_input_text[_i + _max_length])

_prepared_sentences_length = len(_prepared_sentences)

print('_prepared_sentences_length = ', _prepared_sentences_length)
print('Example of prepared sentences')
print(_prepared_sentences[:5])

#create 2 numpy zero filled arrays to prepair for encoding 
_X_train = np.zeros((_prepared_sentences_length, _max_length, _sorted_characters_length), dtype=np.bool)
_Y_train = np.zeros((_prepared_sentences_length, _sorted_characters_length), dtype=np.bool)
#enumerate each _perpared_sentences in to char and numbers for training 
for _i, _s in enumerate(_prepared_sentences):
    for _a, char in enumerate(_s):
        _X_train[_i, _a, _words_to_index[char]] = 1
    _Y_train[_i, _words_to_index[_prepared_char[_i]]] = 1

#keras sequential model
model = Sequential()
#LSTM layer with 128 nodes and an input shape(60,45)
model.add(LSTM(128, input_shape=(_max_length, _sorted_characters_length)))
#dropout layer to reduce overfitting
model.add(Dropout(0.2))
#the dense layer is comprised of 45 nodes
model.add(Dense(len(_sorted_characters), activation='softmax'))
#print model summary
model.summary()
#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

#testing callback functions. The function OnEpochEnd is call at the end of each epoch
def OnEpochEnd(epoch, _):
    print('Do Something')

_epoch_test = LambdaCallback(on_epoch_end=OnEpochEnd)
#fit the model
model.fit(_X_train, _Y_train, batch_size=128, epochs=64, callbacks=[_epoch_test])


def TestLSTM(_model):
    #create a start point by way of random number to select ramdom text from _input_text
    _start_prediction = random.randint(0, _length_of_input_text - _max_length - 1)
    #select the text
    _prediction_text = _input_text[_start_prediction: _start_prediction + _max_length]

    print('\nRandom sentence as seed is -- ',_prediction_text)
    #prediction loop    
    for _x in range(200):
        
        _temp = 0.2
        
        #create a zero filled numpy array 
        _X_pred = np.zeros((1, _max_length, _sorted_characters_length), dtype=np.bool)
        #convert the _perdiction_text into numbers
        for _v, _c in enumerate(_prediction_text):
            _X_pred[0, _v, _words_to_index[_c]] = 1
        
        #the prdiction is retuirned and converted to an numpy array of float 64
        _prediction = np.asarray(_model.predict(_X_pred, verbose=0)[0]).astype('float64')
        #create the probabilty for the next character binomial distribution
        _prediction = np.log(_prediction) / _temp
        _prediction_exponent = np.exp(_prediction)
        _prediction = _prediction_exponent / np.sum(_prediction_exponent)
        #one hot encoded of the highest probility for the next char
        _probability = np.random.multinomial(1, _prediction, 1)
        #return the index with the value of 1
        _next_index = np.argmax(_probability)
        #return the character with according the index from _next_index
        _next_char = _index_to_words[_next_index]
        #build the string into senetence
        _prediction_text = _prediction_text[1:] + _next_char

    return _prediction_text


for _p in range(10):
    print('\nPrediction number ', _p)
    print('\nPredicted Sentence is -- ', TestLSTM(model))


