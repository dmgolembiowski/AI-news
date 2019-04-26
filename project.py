#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:01:56 2019

@authors: Jeffrey Borovac, David Golembiowski, Sydney Leither

Based on
https://towardsdatascience.com/writing-like-shakespeare-with-machine-learning-in-pytorch-d77f851d910c
"""
# Importing libraries
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

# Make vectors in the form [0,0,1,0] to represent dictonary
def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot
    
# Defining method to make mini-batches for training
def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total #floor int divison //
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

# Declaring the model
class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=512, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.words = tokens
        global int2word
        global word2int
        
        self.int2word = int2word
        self.word2int = word2int
        
        #define the LSTM
        self.lstm = nn.LSTM(len(self.word2int), n_hidden, n_layers, #first int in m2
                            dropout=drop_prob, batch_first=True)
        
        #define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        #define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, len(self.int2word))
      
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
        #print("forward", x.size())
        #get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        
        #pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)
        
        #put x through the fully-connected layer
        out = self.fc(out)
        
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden
       
# Declaring the train method
def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if(train_on_gpu):
        net.cuda()
    
    counter = 0
    n_words = len(net.int2word)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            
            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_words)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs, h)
            
            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_words)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
                      
# Defining a method to generate the next character
def predict(net, word, h=None, top_k=None): 
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        # tensor inputs
        x = np.array([[net.word2int[word]]])
        x = one_hot_encode(x, len(net.int2word))
        inputs = torch.from_numpy(x)
        
        if(train_on_gpu):
            inputs = inputs.cuda()
        
        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h)

        # get the character probabilities
        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.int2word))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        word = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return net.int2word[word], h
        
# Declaring a method to generate new text
def sample(net, size, prime, top_k=None):
    def sanitize(_text):
        _text = _text.strip("\n")
        _text = _text.strip('\\')
        return _text
    
    currentIndex = 2
    for i in range(10):
        article = "article" + str(i) + ".txt"
        # Open shakespeare text file and read in data as `text`
        wordSet = set()
        with open(article, 'r') as f:
            text = f.read()
            # We create two dictionaries:
            # 1. int2word, which maps integers to characters
            # 2. word2int, which maps characters to integers
            #text = [sanitize(t)+' ' for t in text.split(" ") if t]
            text = [t+' ' for t in text.split(" ") if t]
            text = set(text)
            #wordDifference = text - wordSet
            wordDifference = wordSet.symmetric_difference(text) - wordSet.difference(text)
            wordSet = wordSet.union(text)
            words = tuple(wordSet)#need to save outside loop??
            global int2word
            global word2int
            #enumeratedWords = OrderedDict(enumerate(words, len(word2int)))
            #enumeratedWords2 = enumeratedWords#does update change its parameter
            #int2word.update(enumeratedWords)#TODO each update is not increasing the dictionaries by the same amount
            #word2int.update({word: ii for ii, word in enumeratedWords2.items()})
            for w in wordDifference:
                if w in word2int.keys():
                    continue
                int2word[currentIndex] = w
                word2int[w] = currentIndex
                currentIndex += 1
            print(len(int2word), len(word2int))
            #int2word = OrderedDict(enumerate(words))
            #word2int = {word: ii for ii, word in int2word.items()} #reverse mapping from character to int
    
    for i in range(10):
        article = "article" + str(i) + ".txt"
        # Open shakespeare text file and read in data as `text`
        #wordSet = set()
        with open(article, 'r') as f:
            text = f.read()
            text = [t+' ' for t in text.split(" ") if t]
            # We create two dictionaries:
            # 1. int2word, which maps integers to characters
            # 2. word2int, which maps characters to integers
            #wordSet = set(text).union(wordSet)
            #words = tuple(wordSet)#need to save outside loop??
            #global int2word
            #global word2int
            # Encode the text
            encoded = np.array([word2int[word] for word in text])
            if net == None:
                net = CharRNN(words, n_hidden, n_layers)
                if(train_on_gpu):
                    net.cuda()
                else:
                    net.cpu()
            #net.lstm.input_size = len(int2word)
            net.words = words
            # train the model
            train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=50)

    net.eval() # eval mode
    
    # First off, run through the prime characters
    #words = [word for word in prime]
    words = []
    words.append(prime)
    h = net.init_hidden(1)
    #for word in prime:
    word, h = predict(net, words[-1], h, top_k=top_k)

    #words.append(word)
    words.append(word)
    
    # Now pass in the previous character and get a new one
    for ii in range(size): 
        word, h = predict(net, words[-1], h, top_k=top_k)
        words.append(word)

    return ''.join(words)

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')
    
# Define and print the net
n_hidden=512
n_layers=2

# Declaring the hyperparameters
batch_size = 128
seq_length = 100
n_epochs = 10 # start smaller if you are just testing initial behavior
    
# Generating new text
net = None
int2word = {}
word2int = {}
print(sample(net, 1000, prime='a ', top_k=100))
