'''
tanh(·) is selected as the activation function among the input layer and the hidden layers.
Between the last hidden layer and the output layer, the identity activation function is used.
'''
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY('SC')
class SC_DNN(nn.Module):
    def __init__(self,args):
        super().__init__()
        dnn_seq = []
        input_size = args.input_size
        for hidden_size in args.dnn_hidden_sizes:
            dnn_seq.append(nn.Linear(input_size, hidden_size))
            dnn_seq.append(nn.ReLU())
            input_size = hidden_size
        dnn_seq.append(nn.Linear(input_size,1))
        self.dnn = nn.Sequential(*dnn_seq)

        self.cls_model = LogisticRegression(class_weight='balanced',)

    def forward(self,x):
        return self.dnn(x).squeeze(-1)

    def fit(self,hi,y):
        self.cls_model.fit(hi.reshape(-1, 1),y)

    def predict(self,X):
        hi = self.forward(X).detach()
        return self.cls_model.predict(hi.reshape(-1, 1))