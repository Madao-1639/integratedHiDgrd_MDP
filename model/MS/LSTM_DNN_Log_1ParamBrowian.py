import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.LSTM_DNN_1ParamBrowian import BaseRTF
from .CustomFlexCoef import select_flex_coef
from loss import _loss_reduction

class LLT(nn.Module):
    '''Learnable Logarithm Transformer'''
    def __init__(self,phi=9,_delta=1e-7):
        super().__init__()
        self.phi = nn.Parameter(torch.FloatTensor([phi]))
        self.delta = _delta
    def forward(self,x):
        return torch.log(F.relu(x+self.phi)+self.delta)

class MSRTF(BaseRTF):
    '''Multi-Stage model for RTF dataset'''
    def __init__(self,phi=9,_delta=1e-7,**Base_kwargs):
        super().__init__(**Base_kwargs)
        self.fix_point = Base_kwargs['args'].MS_fix_point
        self.drop_first = Base_kwargs['args'].MS_drop_first
        self.get_flex_coef = select_flex_coef(Base_kwargs['args'].MS_flex_type)
        self.hi_transformer = LLT(phi,_delta)

    def mfe_loss(self,x,UUT, reduction = 'mean'):
        if self.fix_point is not None:
            x = self.hi_transformer(x[self.fix_point-1:])
            return super().mfe_loss(x,UUT, reduction)
        else:
            x = x[self.drop_first:]
            reduction = 'sum' if reduction == 'mean' else reduction
            loss = self.get_flex_coef(x)@super().mfe_loss(self.hi_transformer(x),UUT, reduction='none')
            return _loss_reduction(loss,reduction)

    @torch.no_grad
    def transform_deg_hi(self,hi,_epsilon=0):
        if self.fix_point is not None:
            turning_index = self.fix_point-1
        else:
            hi = hi[self.drop_first:]
            flex_coef = self.get_flex_coef(hi)
            turning_index = torch.min(torch.argwhere(flex_coef >= _epsilon))
        return self.hi_transformer(hi[turning_index:])