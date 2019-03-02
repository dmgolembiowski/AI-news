#!/usr/bin/env python3
import numpy as np
from __init__ import SydneyClass

class InputGate:

    def __init__(self, position, value, input_Legal=True, is_Open=False):
        self.position = position
        self.value = value
        self.input_Legal = input_Legal
        self.is_Open = is_Open 
        self.correspondence = {self.position : self.value}

    def _input(self, new_Value):
        self.value = new_Value
        self.correspondence[self.value] = self.value


class OutputGate:

    def __init__(self, position, value, input_Legal=False, is_Open=True):
        self.position = position
        self.value = value
        self.input_Legal = input_Legal
        self.is_Open = is_Open
        self.activation_Threshold = 1

class ForgetGate:
    Forget_Threshold = np.Inf

    def __init__(self, position, value, input_Legal=True):
        self.position = position
        self.value = value
        self.input_Legal = input_Legal

    def activation(self):
        return 

class ConstantErrorCarousel:
    Global_Error_Flow = 1
    Carousel_Riders = []
    cec = {}

    def __init__(self, error_Signals, unit):
        self.error_Signals = error_Signals
        self.unit = unit
        ConstantErrorCarousel.Carousel_Riders.append(self.__repr__)
        ConstantErrorCarousel.auto_get(self, len(ConstantErrorCarousel.Carousel_Riders)-1)

    def as_dict(self):
        return self.__dict__

    def auto_get(self, key):
        ConstantErrorCarousel.cec[key] = ConstantErrorCarousel.as_dict(self)

    def __repr__(self):
        return 'Carousel_Rider(%s)' % ", ".join(k+"="+repr(v) for k,v in self.__dict__.items())

    @classmethod
    def global_Error(cls, q_Value):
        pass

class MemoryCell(SydneyClass):
    """
    LSTM.MemoryCell should be initialized in a non-traditional fashion, accepting
    __init__ arguments of the form `Classname(inputgate, CEC, etc...)` since
    multiple classes share similar attribute names, and this unconventional 
    class instantiation process ensures that MemoryCell instances will not 
    assign the same name to multiple distinct attributes.

    Example Usage:

    

    """
    Cells = {}

    def __init__(self, sentenceParams, wordParams, 
            linear_Unit, inputGate, outputGate, forgetGate,
            constantErrorCarousel):
        super().__init__(sentenceParams, wordParams)
        self.linear_Unit = linear_Unit
        self.inputGate = inputGate # := InputGate(*args)
        self.outputGate = outputGate # := OutputGate(*args)
        self.forgetGate = forgetGate # := ForgetGate(*args)
        self.constantErrorCarousel = constantErrorCarousel # := ConstantErrorCarousel(*args)
        self.id = MemoryCell.__repr__(self)
        MemoryCell.Cells[self.id] = self.__dict__

    def __repr__(self):
        return hex(id(self))
