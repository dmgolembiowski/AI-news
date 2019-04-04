#!/usr/bin/env python3
"""
AI-News/learning/vectorization.py 
Modified from GitHub repository source: authored by Igor Dzreyev from https://github.com/joosthub/PyTorchNLPBook
Attribution: "Natural Language Processing with PyTorch by Delip Rao and Brian McMahan (O'Reilly). Copyright 2019, Delip Rao and Brian McMahan, 978-1-491-97823-8"

The source code is rightfully reproduced under the Apache 2.0 License for academic research at Baldwin-Wallace University, Dept. of Computer Science.
"""
import torch
import torch.nn as nn


class ReviewClassifier(nn.Module):
    """ a simple perceptron based classifier """
    def __init__(self, num_features):
        """
        Args:
            num_features (int): the size of the input feature vector
        """
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, 
                             out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        """The forward pass of the classifier
        
        Args:
            x_in (torch.Tensor): an input data tensor. 
                x_in.shape should be (batch, num_features)
            apply_sigmoid (bool): a flag for the sigmoid activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch,)
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out
