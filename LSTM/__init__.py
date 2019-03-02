#!/usr/bin/env python3
import numpy as np
from LSTM import InputGate, OutputGate, ForgetGate, ConstantErrorCarousel, MemoryCell

class SydneyClass:
  def __init__(self, wordParams, sentenceParams):
    self.wordParams = wordParams
    self.sentenceParams = sentenceParams
