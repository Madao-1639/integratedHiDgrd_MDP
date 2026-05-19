import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam, SGD
from utils.registry import OPTIMIZER_REGISTRY

OPTIMIZER_REGISTRY.register("Adam", Adam)
OPTIMIZER_REGISTRY.register("SGD", SGD)

def _loss_reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    '''Apply reduction to a loss tensor.

    Args:
        loss (torch.Tensor): The loss tensor to reduce.
        reduction (str): Reduction mode. One of 'none', 'mean', 'sum'.

    Returns:
        torch.Tensor: The reduced loss tensor.
    '''
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
    '''Compute Focal Loss for binary classification.

    Args:
        logits (torch.Tensor): Unnormalized predictions.
        y_true (torch.Tensor): Ground truth labels.
        alpha (float): Balancing factor. Defaults to 0.25.
        gamma (float): Focusing parameter. Defaults to 2.
        reduction (str): Reduction mode. Defaults to "none".

    Returns:
        torch.Tensor: The focal loss tensor.
    '''
    ce_loss = F.binary_cross_entropy_with_logits(logits, y_true, reduction="none")
    y_pred = F.sigmoid(logits)
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if 0 < alpha < 1:
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * loss
    return _loss_reduction(loss, reduction)

class MFELoss_1ParamBrownian(nn.Module):
    '''MFE loss based on 1-parameter Brownian motion model.'''

    def __init__(self, n_UUT: int, mu0: float = 1, sigma0: float = 1, sigma_square: float = 1) -> None:
        '''Initialize the MFE loss with learnable per-UUT parameters.

        Args:
            n_UUT (int): Number of units.
            mu0 (float): Mean for parameter initialization. Defaults to 1.
            sigma0 (float): Std for parameter initialization. Defaults to 1.
            sigma_square (float): Initial value for sigma_square. Defaults to 1.
        '''
        super().__init__()
        self.theta_train = nn.Parameter(torch.empty(n_UUT).normal_(mu0, sigma0))
        self.sigma_square = nn.Parameter(torch.FloatTensor([sigma_square]))

    def forward(self, X, indice, reduction='mean'):
        theta = self.theta_train[indice].unsqueeze(1)
        loss = torch.log(self.sigma_square + 1e-7) + \
            (X.diff(prepend=torch.zeros(
                X.shape[0], 1, *X.shape[2:],
                dtype=X.dtype, 
                device=X.device
            )) - theta).square()
        return _loss_reduction(loss, reduction)

class MFELoss_LSTM(nn.Module):
    '''MFE loss with LSTM-based degradation model.'''

    def __init__(self, n_UUT: int, lstm_hidden_size: int, num_lstm_layers: int = 1, lstm_dropout: float = 0.0, mu0: float = 1, sigma0: float = 1) -> None:
        '''Initialize the LSTM-based MFE loss.

        Args:
            n_UUT (int): Number of units.
            lstm_hidden_size (int): Hidden size of the LSTM.
            num_lstm_layers (int): Number of LSTM layers. Defaults to 1.
            lstm_dropout (float): Dropout rate for LSTM. Defaults to 0.0.
            mu0 (float): Mean for parameter initialization. Defaults to 1.
            sigma0 (float): Std for parameter initialization. Defaults to 1.
        '''
        super().__init__()
        self.tmax = 500
        self.lstm = nn.LSTM(
            1, lstm_hidden_size, num_lstm_layers,
            dropout=lstm_dropout, batch_first=True
        )
        self.Gamma_train = nn.Parameter(torch.empty(n_UUT, lstm_hidden_size).normal_(mu0, sigma0))

    def forward(self, hi, t, indice, lengths, reduction='mean'):
        Gamma = self.Gamma_train[indice]
        t = t.unsqueeze(-1) / self.tmax
        t = pack_padded_sequence(t, lengths, batch_first=True, enforce_sorted=False)
        psi, (_, _) = self.lstm(t)
        psi, lengths = pad_packed_sequence(psi, batch_first = True)
        dgrd_status = (psi@Gamma.unsqueeze(-1)).squeeze(-1)
        loss = (hi - dgrd_status).square()
        return _loss_reduction(loss, reduction)

def MVFLoss(hi_f: torch.Tensor, m: float = 1, reduction: str = "none") -> torch.Tensor | float:
    '''Compute Minimum Variance Fidelity loss.

    Args:
        hi_f (torch.Tensor): Health indicator values.
        m (float): Target value. Defaults to 1.
        reduction (str): Reduction mode. Defaults to "none".

    Returns:
        torch.Tensor | float: The MVF loss.
    '''
    loss = torch.square(hi_f - m)
    return _loss_reduction(loss, reduction)

def MONLoss(hi_pre: torch.Tensor, hi_cur: torch.Tensor | None = None, c: float = 0, penalty_type: str = 'exp', reduction: str = "none") -> torch.Tensor | float:
    '''Compute Monotonicity loss penalizing non-monotonic degradation.

    Args:
        hi_pre (torch.Tensor): Previous health indicator values.
        hi_cur (torch.Tensor | None): Current health indicator values. If None, computes from hi_pre.diff(). Defaults to None.
        c (float): Offset constant. Defaults to 0.
        penalty_type (str): Penalty function type. One of 'linear', 'exp', 'tanh'. Defaults to 'exp'.
        reduction (str): Reduction mode. Defaults to "none".

    Returns:
        torch.Tensor | float: The monotonicity loss.
    '''
    if hi_cur is not None:
        diff = hi_pre - hi_cur + c 
    else:
        # Run-to-failure
        diff = -hi_pre.diff() + c
    if penalty_type == 'linear':
        inner_term = diff
    elif penalty_type == 'exp':
        inner_term = torch.exp(diff) - 1
    elif penalty_type == 'tanh':
        inner_term = torch.tanh(diff)
    else:
        raise ValueError(
            f"Invalid Value for arg 'penalty_type': '{penalty_type} \n Supported reduction modes: 'linear', 'exp', 'tanh'"
        )
    loss = F.relu(inner_term)
    return _loss_reduction(loss, reduction)

def CONLoss(hi_ppre: torch.Tensor, hi_pre: torch.Tensor | None = None, hi_cur: torch.Tensor | None = None, **mon_kwargs) -> torch.Tensor | float:
    '''Compute Concavity loss penalizing non-concave degradation.

    Args:
        hi_ppre (torch.Tensor): Pre-previous health indicator values.
        hi_pre (torch.Tensor | None): Previous health indicator values. Defaults to None.
        hi_cur (torch.Tensor | None): Current health indicator values. Defaults to None.
        **mon_kwargs: Keyword arguments passed to MONLoss.

    Returns:
        torch.Tensor | float: The concavity loss.
    '''
    if hi_pre is not None:
        d_pre = hi_ppre - hi_pre
        d_cur = hi_pre - hi_cur
        return MONLoss(d_pre, d_cur, **mon_kwargs)
    else:
        return MONLoss(hi_ppre.diff(), **mon_kwargs)