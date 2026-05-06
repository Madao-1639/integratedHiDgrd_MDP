import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam, SGD
from utils.registry import OPTIMIZER_REGISTRY

OPTIMIZER_REGISTRY.register("Adam", Adam)
OPTIMIZER_REGISTRY.register("SGD", SGD)

def _loss_reduction(loss,reduction: str):
    if reduction == "none":
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

def FocalLoss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    ce_loss = F.binary_cross_entropy_with_logits(logits, y_true, reduction="none")
    y_pred = F.sigmoid(logits)
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if 0 < alpha < 1:
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * loss
    return _loss_reduction(loss,reduction)

class MFELoss_1ParamBrownian(nn.Module):
    def __init__(self, n_UUT, mu0=1, sigma0=1, sigma_square=1):
        super().__init__()
        self.theta_train = nn.Parameter(torch.empty(n_UUT).normal_(mu0,sigma0))
        self.sigma_square = nn.Parameter(torch.FloatTensor([sigma_square]))

    def forward(self, X, indice, reduction = 'mean'):
        theta = self.theta_train[indice].unsqueeze(1)
        loss = torch.log(self.sigma_square + 1e-7) + \
            (X.diff(prepend=torch.zeros(
                X.shape[0], 1, *X.shape[2:],
                dtype=X.dtype, 
                device=X.device
            )) - theta).square()
        return _loss_reduction(loss, reduction)

class MFELoss_LSTM(nn.Module):
    def __init__(self, n_UUT, 
                lstm_hidden_size, num_lstm_layers = 1, lstm_dropout = 0.0,
                mu0 = 1, sigma0 = 1):
        super().__init__()
        self.tmax = 500
        self.lstm = nn.LSTM(
            1, lstm_hidden_size, num_lstm_layers,
            dropout = lstm_dropout, batch_first = True
        )
        self.Gamma_train = nn.Parameter(torch.empty(n_UUT,lstm_hidden_size).normal_(mu0,sigma0))

    def forward(self, hi, t, indice, lengths, reduction = 'mean'):
        Gamma = self.Gamma_train[indice]
        t = t.unsqueeze(-1) / self.tmax
        t = pack_padded_sequence(t, lengths, batch_first = True, enforce_sorted = False)
        psi,(_,_) = self.lstm(t)
        psi, lengths = pad_packed_sequence(psi, batch_first = True)
        dgrd_status = (psi@Gamma.unsqueeze(-1)).squeeze(-1)
        loss = (hi - dgrd_status).square()
        return _loss_reduction(loss, reduction)

def MVFLoss(hi_f,m=1,reduction="none",):
    loss = torch.square(hi_f-m)
    return _loss_reduction(loss,reduction)

def MONLoss(hi_pre,hi_cur=None,c=0,penalty_type='exp',reduction="none",):
    if hi_cur is not None:
        diff = hi_pre - hi_cur + c 
    else:
        # Run-to-failure
        diff = -hi_pre.diff() + c
    if penalty_type == 'linear':
        inner_term = diff
    elif penalty_type == 'exp':
        inner_term = torch.exp(diff)-1
    elif penalty_type == 'tanh':
        inner_term = torch.tanh(diff)
    else:
        raise ValueError(
            f"Invalid Value for arg 'penalty_type': '{penalty_type} \n Supported reduction modes: 'linear', 'exp', 'tanh'"
        )
    loss = F.relu(inner_term)
    return _loss_reduction(loss,reduction)

def CONLoss(hi_ppre,hi_pre=None,hi_cur=None,**mon_kwargs):
    if hi_pre is not None:
        d_pre = hi_ppre - hi_pre
        d_cur = hi_pre - hi_cur
        return MONLoss(d_pre,d_cur,**mon_kwargs)
    else:
        return MONLoss(hi_ppre.diff(),**mon_kwargs)