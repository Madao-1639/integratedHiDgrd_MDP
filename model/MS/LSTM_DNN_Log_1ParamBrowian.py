import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.LSTM_DNN_1ParamBrowian import BaseRTF
from loss import _loss_reduction

def select_flex_coef(flex_type,**kw_flex):
    if flex_type == 'ReLULBias':
        return ReLULBias(**kw_flex)



class ReLULBias(nn.Module):
    '''ReLU with a learnable bias'''
    def __init__(self,bias=10):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor([bias]))
    def forward(self,x):
        return F.relu(x+self.bias)

class LLT(nn.Module):
    '''Learnable Logarithm Transformer'''
    def __init__(self,phi=10,_delta=1e-7):
        super().__init__()
        self.phi = nn.Parameter(torch.FloatTensor([phi]))
        self.delta = _delta
    def forward(self,x):
        return torch.log(F.relu(x+self.phi)+self.delta)

class MSRTF(BaseRTF):
    '''Multi-Stage model for RTF dataset'''
    def __init__(self,phi=10,_delta=1e-7,**Base_kwargs):
        super().__init__(**Base_kwargs)
        self.fix_point = Base_kwargs['args'].MS_fix_point
        self.get_flex_coef = select_flex_coef(Base_kwargs['args'].MS_flex_type)
        self.hi_transformer = LLT(phi,_delta)

    def mfe_loss(self,x,UUT, reduction = 'mean'):
        if self.fix_point is not None:
            x = self.hi_transformer(x[self.fix_point-1:])
            return super().mfe_loss(x,UUT, reduction)
        else:
            reduction = 'sum' if reduction == 'mean' else reduction
            x = self.hi_transformer(x)
            loss = self.get_flex_coef(x)@super().mfe_loss(x,UUT, reduction='none')
            return _loss_reduction(loss,reduction)

    @torch.no_grad
    def transform_deg_hi(self,hi,_epsilon=0):
        if self.fix_point is not None:
            turning_index = self.fix_point-1
        else:
            flex_coef = self.get_flex_coef(hi)
            turning_index = torch.min(torch.argwhere(flex_coef >= _epsilon))
        return self.hi_transformer(hi[turning_index:])