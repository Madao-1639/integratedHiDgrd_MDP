import torch
import torch.nn as nn
from ..CustomActFunc import select_activate
from loss import _loss_reduction


class BaseRTF(nn.Module):
    def __init__(self, args, train_UUTs, mu0=1, sigma0=1, sigma_square=1):
        super().__init__()
        self.lstm = nn.LSTM(
            args.input_size, args.lstm_hidden_size, args.num_lstm_layers,
            dropout=args.lstm_dropout, batch_first=True
            )
        dnn_seq = []
        input_size = args.lstm_hidden_size
        for hidden_size in args.dnn_hidden_sizes:
            dnn_seq.append(nn.Linear(input_size, hidden_size))
            dnn_seq.append(nn.ReLU())
            input_size = hidden_size
        dnn_seq.append(nn.Linear(input_size,1))
        self.dnn = nn.Sequential(*dnn_seq)
        self.activate = select_activate(args.activate)
        self.cls_thres = args.cls_thres

        # 1ParamBrownian
        self.train_UUT_dict = {UUT:i for i,UUT in enumerate(train_UUTs)}
        self.theta_train = nn.Parameter(torch.empty(len(train_UUTs)).normal_(mu0,sigma0))
        self.sigma_square = nn.Parameter(torch.FloatTensor([sigma_square]))

    def forward(self,x,hidden=None):
        lstm_output, (h,c) = self.lstm(x,hidden)
        hi = self.dnn(lstm_output.squeeze(0)).squeeze(-1)
        p = self.activate(hi)
        return hi, p
    
    def predict(self,x,thres=None,**fw_kwargs):
        hi, p = self.forward(x,**fw_kwargs)
        if not thres:
            thres = self.cls_thres
        y_pred = (p>=thres).detach()
        return y_pred

    def mfe_loss(self,x,UUT, reduction = 'mean'):
        idx = self.train_UUT_dict[UUT]
        theta = self.theta_train[idx]
        loss = torch.log(self.sigma_square + 1e-7) + \
            (x.diff(prepend=torch.FloatTensor([0])) - theta).square()
        return _loss_reduction(loss, reduction)

    @torch.no_grad()
    def fit_prior_dist(self):
        self.mu0 = self.theta_train.mean().item()
        self.sigma0_square = self.theta_train.var().item()

    @torch.no_grad()
    def update_posterior_dist(self,k,lk):
        sigma1_square = 1/((1/self.sigma0_square)+(k/self.sigma_square.item()))
        mu1 = sigma1_square*((self.mu0/self.sigma0_square)+(lk/self.sigma_square.item()))
        return mu1, sigma1_square


class BaseTW(BaseRTF):
    def forward(self,x,hidden=None):
        '''Input: x (batch_size, window_width, in_features)
        Output:
            hi (batch_size,) -> Health index.
            p (batch_size,) -> Failure probability.'''
        lstm_output, (h,c) = self.lstm(x,hidden)
        hi = self.dnn(h.squeeze(0)).squeeze(-1)
        p = self.activate(hi)
        return hi, p
    
    def _UUT2idx(self,UUT):
        if hasattr(UUT,'__getitem__'): # The passed UUT is a sequence
            return [self.train_UUT_dict[_] for _ in UUT]
        else:
            return self.train_UUT_dict[UUT]

    def mfe_loss(self,hi_pre,hi_cur,UUT, reduction = 'none'):
        # Calculate Model Fitting Error(MFE) loss by MLE
        indice = self._UUT2idx(UUT)
        theta = self.theta_train[indice]
        loss = torch.log(self.sigma_square) + torch.square(hi_cur - hi_pre - theta)/self.sigma_square
        return _loss_reduction(loss,reduction)