#!/usr/bin/env python3

import torch
import torch.nn as nn # Neural Net
import data 
from GRU import Update, Reset
from learning import 

def main(trainFirst=False, ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if trainFirst:
        newsGenerator = data.main()

