import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY('Base')
class Base(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lstm = nn.LSTM(
            args.input_size, args.lstm_hidden_size, args.num_lstm_layers,
            dropout=args.lstm_dropout, batch_first = True,
            )
        dnn_seq = []
        input_size = args.lstm_hidden_size
        for hidden_size in args.dnn_hidden_sizes:
            dnn_seq.append(nn.Linear(input_size, hidden_size))
            dnn_seq.append(nn.ReLU())
            input_size = hidden_size
        dnn_seq.append(nn.Linear(input_size,1))
        self.dnn = nn.Sequential(*dnn_seq)
        self.cls_thres = args.cls_thres
    def forward(self, X, lengths, hidden = None):
        X = pack_padded_sequence(X, lengths, batch_first = True, enforce_sorted = False)
        lstm_output, (h, c) = self.lstm(X, hidden)
        lstm_output, lengths = pad_packed_sequence(lstm_output, batch_first = True)
        hi = self.dnn(lstm_output).squeeze(-1) # (batch_size, max_len)
        mask = torch.arange(hi.shape[1]) < lengths.unsqueeze(1)
        # hi *= mask
        return hi, mask
    
    def predict(self, X, lengths, thres=None):
        hi, mask = self.forward(X, lengths)
        if not thres:
            thres = self.cls_thres
        p = F.sigmoid(hi)
        Y_pred = (p >= thres).detach()
        return Y_pred, mask