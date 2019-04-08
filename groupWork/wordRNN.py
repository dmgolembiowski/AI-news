#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:01:56 2019

@authors: Jeffrey Borovac, David Golembiowski, Sydney Leither

Based on
https://towardsdatascience.com/writing-like-shakespeare-with-machine-learning-in-pytorch-d77f851d910c
and
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

with open('article.txt', 'r') as file:
    text = file.read()
    
text = text.split(" ")
words = tuple(set(text))
int2word = dict(enumerate(words))
word2int = {word: ii for ii, word in int2word.items()} #reverse mapping from character to int

encoded = np.array([word2int[word] for word in text])

# vectors with all zeros except for a 1 at a selected int
# takes features and makes them binary
def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arrange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
""" old RNN
class WordRRN(nn.Module):
    def __init__(self, tokens, drop_prob=0.5):
        super().__init__()
        
        # create word dictionaries
        self.words = tokens
        self.int2word = dict(enumerate(self.chars))
        self.word2int = {word: ii for ii, word in self.int2word.items()}
        
        # define GRU TODO
        self.gru = nn.GRU()
        
        self.dropout = nn.Dropout(drop_prob)
        
        # output layer TODO
        # self.fc = ...

        
    # move forward through the network TODO
    def forward(self, x, hidden):
        r_output, hidden = self.gru(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        #out = self.fc(out)
            
        #move backwards? TODO
        # ...
"""
"""
    # defines an optimizer (Adam), loss, and does validation data and inits hidden state of RNN
    # forward and backpropagation
    def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=01, print_every=10):
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        #training and validation data
"""