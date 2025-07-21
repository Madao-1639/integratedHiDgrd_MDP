import torch
import torch.nn as nn
import torch.nn.functional as F

def select_flex_coef(flex_type,**kw_flex):
    if flex_type == 'SReLU':
        return SReLU(**kw_flex)
    elif flex_type == 'LSReLU':
        return LSReLU(**kw_flex)
    elif flex_type == 'SLSReLU':
        return SLSReLU(**kw_flex)

class SReLU(nn.Module):
    '''Shifted ReLU'''
    def __init__(self, bias=8.048):
        super().__init__()
        self.bias = torch.FloatTensor([bias])

    def forward(self, x):
        return F.relu(x + self.bias)

class LSReLU(SReLU):
    '''Learnable SReLU'''
    def __init__(self,bias=8.048):
        super().__init__(bias)
        self.bias = nn.Parameter(self.bias)

class SLSReLU(LSReLU):
    '''Scaled Learnable Shifted ReLU'''
    def forward(self,x):
        return 1-torch.exp(-super().forward(x))