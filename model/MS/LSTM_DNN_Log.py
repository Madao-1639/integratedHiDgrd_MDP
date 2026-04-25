import torch
from torch import nn
import torch.nn.functional as F
from ..base.LSTM_DNN import Base
from utils.registry import MODEL_REGISTRY

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

@MODEL_REGISTRY('MS')
class MS(Base):
    '''Multi-Stage model'''
    def __init__(self, args):
        super().__init__(args=args)
        self.fix_point = args.MS_fix_point
        self.drop_first = args.MS_drop_first
        self.hi_transformer = LLT(c1 = args.MS_LLT_c1, c2 = args.MS_LLT_c2)
    
    def forward(self, X, lengths, hidden = None):
        logits, cls_mask = super().forward(X, lengths, hidden)
        mfe_mask = torch.detach_copy(cls_mask)
        if self.fix_point:
            hi = self.hi_transformer(logits[:, self.fix_point - 1:])
            mfe_mask = mfe_mask[:, self.fix_point - 1:]
        else:
            hi = self.hi_transformer(logits[:, self.drop_first:])
            mfe_mask = mfe_mask[:, self.drop_first:]
        return logits, hi, cls_mask, mfe_mask