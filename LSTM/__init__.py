#!/usr/bin/env python3
import numpy as np
#from LSTM import InputGate, OutputGate, ForgetGate, ConstantErrorCarousel, MemoryCell
from . import LSTM
InputGate = LSTM.InputGate
OutputGate = LSTM.OutputGate
ForgetGate = LSTM.ForgetGate
ConstantErrorCarousel = LSTM.ConstantErrorCarousel
MemoryCell = LSTM.MemoryCell

class GarfieldClass:
  def __init__(self, wordParams, sentenceParams):
    self.wordParams = wordParams
    self.sentenceParams = sentenceParams
