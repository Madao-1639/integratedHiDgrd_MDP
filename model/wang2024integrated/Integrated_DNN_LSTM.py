import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY('Integrated')
class Integrated_DNN_LSTM(nn.Module):
    '''
    The main difference between this model and SC_DNN is that the activation function is softplus instead of tanh.
    '''

    def __init__(self, args):
        super().__init__()
        dnn_seq = []
        input_size = args.input_size
        for hidden_size in args.dnn_hidden_sizes:
            dnn_seq.append(nn.Linear(input_size, hidden_size))
            dnn_seq.append(nn.Softplus())
            input_size = hidden_size
        dnn_seq.append(nn.Linear(input_size,1))
        self.dnn = nn.Sequential(*dnn_seq)

        self.cls_model = LogisticRegression(class_weight='balanced',)

    def forward(self,x):
        return self.dnn(x).squeeze(-1)  # Degradation signals / HI

    def fit(self,hi,y):
        self.cls_model.fit(hi.reshape(-1, 1),y)

    def predict(self,X):
        hi = self.forward(X).detach()
        return self.cls_model.predict(hi.reshape(-1, 1))