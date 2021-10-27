import torch
import torch.nn as nn 

from ...modules import WithRecorder

class Identity(WithRecorder):
    """
    Do nothing.
    """
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, features):
        return features