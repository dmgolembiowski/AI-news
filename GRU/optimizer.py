#!/usr/bin/env python3
import torch 
import torch.nn as nn
import torch.optim as optim


class Perceptron(nn.Module):
    """
    (Documentation Incomplete)
    One linear layer within the model
    """
    def __init__(self, input_dimension):
        super().__init__()
        #self.fc1 = nn.GRUCell
class Optimizer:
    def __init__(self, input_dim, learning_rate):
        self.input_dim = input_dim
        self.learning_rate = learning_rate



