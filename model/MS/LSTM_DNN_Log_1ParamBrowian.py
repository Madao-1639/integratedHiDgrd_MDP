import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.LSTM_DNN_1ParamBrowian import BaseRTF
from .CustomFlexCoef import select_flex_coef
from loss import _loss_reduction

class LLT(nn.Module):
    '''Learnable Logarithm Transformer'''
    def __init__(self,c1=10.5,c2=-0.5,_delta=1e-7):
        super().__init__()
        self.c1 = nn.Parameter(torch.FloatTensor([c1]))
        # self.c1 = 8.2
        self.c2 = nn.Parameter(torch.FloatTensor([c2]))
        # self.c2 = 0
        self.delta = _delta
    def forward(self,x):
        return torch.log(F.relu(x+self.c1)+self.delta)+self.c2

class MSRTF(BaseRTF):
    '''Multi-Stage model for RTF dataset'''
    def __init__(self,args,**Base_kwargs):
        super().__init__(args=args,**Base_kwargs)
        self.fix_point = args.MS_fix_point
        self.drop_first = args.MS_drop_first
        self.get_flex_coef = select_flex_coef(args.MS_flex_type)
        self.hi_transformer = LLT()

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
            turning_index = torch.min(torch.argwhere(flex_coef > _epsilon))
        return self.hi_transformer(hi[turning_index:])