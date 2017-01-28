#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import nltk
import csv
import itertools
import numpy as np
import string 

class Process():
    def __init__(self):
        self.unknown = "UNKNOWN_TOKEN"
        self.start = '+'
        self.end = '-'
        
        print "Reading csv file...\n"
        with open("AnHo_words.csv") as f:
            reader = csv.reader(f, skipinitialspace = True)
            sentences = itertools.chain(*reader)
            self.sent_word =  [nltk.word_tokenize(sent) for sent in sentences]

        word = []
        for sent in self.sent_word:
            for x in sent:
                word.append([x])
        word = [s[0].lower() for s in word]
        
        all_word=[]
        for w in word:
            k=1
            for part in all_word:
                if w == part:
                    k=0
            if k==1:
                all_word.append(w)
        word = all_word

        letter = []
        for i in word:
            x = itertools.chain(i)
            y = [k for k in x]
            if (len(y)>2):
                letter.append(y)
                
        for t in np.arange(len(letter)):
            letter[t] = [self.start]+letter[t]+[self.end]
        # index -> letter 26+2+1 = 29
        vocab = '+'+string.ascii_lowercase+"'"+'-'
        self.index_to_letter = dict([(i,l) for i,l in enumerate(vocab)])
        # letter -> letter
        self.letter_to_index = dict([(l,i) for i,l in enumerate(vocab)])
        
        #  Training set  ^^
        self.X_train = [[self.letter_to_index[i] for i in w[:-1]] for w in letter]
        self.Y_train = [[self.letter_to_index[i] for i in w[1:]] for w in letter]  
        
    def giveInput(self):
        return self.X_train, self.Y_train
    
    def Dictionary(self):
        return self.index_to_letter, self.letter_to_index
        
p = Process()
X_train, Y_train = p.giveInput()
print "Example input & output:"
print X_train[0]
print Y_train[0]
print "\nLength of input & output: %i" %len(Y_train)











