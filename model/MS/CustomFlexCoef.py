import torch
import torch.nn as nn
import torch.nn.functional as F

def select_flex_coef(flex_type,**kw_flex):
    if flex_type == 'ReLULBias':
        return ReLULBias(**kw_flex)

class ReLULBias(nn.Module):
    '''ReLU with a learnable bias'''
    def __init__(self,bias=7.8):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor([bias]))
    def forward(self,x):
        return F.relu(x+self.bias)