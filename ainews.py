#!/usr/bin/env python3

import torch
import torch.nn as nn # Neural Net
from data import dbExtract
from GRU import Update, Reset
#from learning import 
import groupWork
from collections import deque
import numpy as np

articles = collections.deque()
wordToInt = {}



def main(trainFirst=False, ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if trainFirst:
        newsGenerator = data.main()
    allNews = dbExtract.extract()
    current_article = []
    i = 0
    max_count = 10
    while i < max_count:
        articles.append(dbExtract.newArticle(allNews))
        if i == max_count:
            break
        else:
            i += 1
    current_article = [x for x in article[-1]]
    size = len(current_article)
    i = 0
    while i < size:
        representations[]
