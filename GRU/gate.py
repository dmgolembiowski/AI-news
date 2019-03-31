#!/usr/bin/env python3
import torch

class Gate:
    
    @staticmethod
    def sigmoid(tensor):
        return torch.nn.Sigmoid()(tensor)
    
    @staticmethod
    def tanh(tensor):
        return torch.nn.Tanh()(tensor)
    
    @staticmethod
    def join(tensor, *tensors):
        _tensors = ()
        _tensors.add(tensor)
        for t in tensors:
            _tensors.add(t)
        return torch.cat(tensors=_tensors)
    
    @staticmethod
    def multiply(tensor, *tensors):
        total = tensor
        for t in tensors:
            total = total * t
        return total

    @staticmethod
    def gate(prev_Output, new_Input):
        prevOutput = prevOutput.weight * prevOutput
        newInput = new_Input.weight * new_Input
        joined = Gate.join(prevOutput, newInput)
        sigmoided = Gate.sigmoid(joined)
        
class Update(Gate):
    @staticmethod
    def gate(prev_Output, new_Input):
        return super().gate(prev_Output, new_Input)
    # Will need to do other things afterwards
    
class Reset(Gate):
    @staticmethod
    def gate(prev_Output, new_Input):
        return super().gate(prev_Output, new_Input)
    # Will need to the the other steps
    
