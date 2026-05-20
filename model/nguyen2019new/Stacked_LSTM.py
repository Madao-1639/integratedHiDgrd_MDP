
import torch
from torch import nn
import torch.nn.functional as F


class StackedLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        lstm_list = []
        input_size = args.input_size
        num_lstm_layers = len(args.lstm_hidden_sizes)
        for i, hidden_size in enumerate(args.lstm_hidden_sizes):
            lstm_list.append(nn.LSTM(
                input_size = input_size, hidden_size = hidden_size,
                dropout = args.lstm_dropout if i + 1 < num_lstm_layers else 0, 
                batch_first = True
            ))
            input_size = hidden_size
        self.lstm_layers = nn.ModuleList(lstm_list)
        self.ffn = nn.Linear(input_size, 1)
        self.cls_thres = args.cls_thres

    def forward(self,x,hidden=None):
        for lstm_layer in self.lstm_layers:
            x, (_, _) = lstm_layer(x)
        logits = self.ffn(x).squeeze(-1)
        return logits
    
    def predict(self,x,thres=None,**fw_kwargs):
        logits = self.forward(x,**fw_kwargs)
        p = F.sigmoid(logits)
        if not thres:
            thres = self.cls_thres
        y_pred = (p>=thres).detach()
        return y_pred