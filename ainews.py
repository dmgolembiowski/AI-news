#!/usr/bin/env python3

import torch
import torch.nn as nn # Neural Net
from data import dbExtract
from GRU import Update, Reset
#from learning import 
#import groupWork
from collections import deque
import numpy as np
import subprocess
# Late night shenanigans
import multiprocessing as mp
from torch.multiprocessing import Process, Pool, set_start_method
import sys
import random



def main(_resumeTraining=False, 
            training_article_range=None, 
            modelFile='', 
            trainingData="./trainingData/"):
    """
    `trained_article_range` expects either none or 2 element list of
    inclusive range for all seen training article .txt files
    (Semipro-tip usage:)

            >>> ainews.main(_resumeTraining=True, training_article_range=[400,500],
            modelFile='rnn_20_epoch.net')

    Training Routine Steps:
    1. Instantiate the Dataset

    2. Instantiate the model

    3. Instantiate the loss function

    4. Instantiate the optimizer

    5. Iterate over the dataset's training partition and update the model parameters

    6. Iterate over the dataset's validation partition and measure the performance

    7. For new articles, repeat steps 50-100 more times.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if _resumeTraining:
            # Create dictionaries to gather training IDs and validation IDs
            lower_bd = training_article_range[0]
            upper_bd = training_article_range[1]
            partition = {'train':[f"id-{n}" for n in range()], 'validation':[]}
            labels = {}


        set_start_method('spawn', force=True)
    except RuntimeError:
        pass


