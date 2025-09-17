import torch

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from utils.data import select_loader
from utils.utils import test4norm



class BaseEvaluator:
    """Base class for evaluators."""
    def __init__(self, args,
                train_data: pd.DataFrame = None, val_data: pd.DataFrame = None,
                ) -> None:
        self.args = args
        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f'Evaluating on {self.device}')
        self.get_loader(train_data,val_data)
        self.get_model()

    def get_loader(self, train_data: "pd.DataFrame", val_data: "pd.DataFrame") -> None:
        self.val_loader = select_loader(val_data,True,self.args)

        if self.args.record_HI:
            if self.args.record_HI == 'train':
                record_HI_data = train_data
            elif self.args.record_HI == 'val':
                record_HI_data = val_data
            else:
                record_HI_data = pd.concat([train_data,val_data])
            if self.args.data_type == 'TW':
                record_HI_data_type = 'RTFTW'
            else:
                record_HI_data_type = 'RTF'
            self.record_HI_loader = select_loader(record_HI_data,False,self.args,record_HI_data_type)
        else:
            self.record_HI_loader = None

    def get_model(self) -> None:
        self.model = torch.load(self.args.load_model_fp, map_location=self.device, weights_only=False)

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()

        # Classification metrics
        all_y_true = []
        all_y_pred = []
        for UUT,t,X,y_true in self.val_loader:
            X = X.to(self.device)
            y_pred = self.model.predict(X)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='binary', zero_division=0)
        metric_result = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
        }
        return metric_result

    @torch.no_grad()
    def record(self) -> dict:
        self.model.eval()
        # Record HI
        hi_dict = {}
        for UUT,t,X,y_true in self.record_HI_loader:
            X = X.to(self.device)
            hi = self.model(X).detach().numpy()
            hi_dict[UUT] = hi
        return hi_dict



class BaseRTFEvaluator(BaseEvaluator):
    '''BaseRTF Model Evaluator'''

    @torch.no_grad()
    def record(self):
        self.model.eval()

        # Record HI
        hi_dict = {}
        for UUT,t,X,y_true in self.record_HI_loader:
            X = X.to(self.device)
            hi,p = self.model(X)
            hi_dict[UUT] = hi.detach().numpy()

        # Test for normality
        nt_summary = test4norm(hi_dict)

        return hi_dict, nt_summary



class MSRTFEvaluator(BaseRTFEvaluator):
    '''MSRTF Model Evaluator'''

    # def get_model(self):
    #     self.model = MSRTF(args=self.args, train_UUTs=[])
    #     del self.model.theta_train
    #     super(BaseRTFEvaluator,self).get_model()

    @torch.no_grad()
    def record(self):
        self.model.eval()

        # Record HI
        hi_dict = {}
        deg_hi_dict = {}
        for UUT,t,X,y_true in self.record_HI_loader:
            X = X.to(self.device)
            hi,p = self.model(X)
            hi_dict[UUT] = hi.detach().numpy()
            deg_hi_dict[UUT] = self.model.transform_deg_hi(hi).detach().numpy()

        # Test for normality
        nt_summary = test4norm(deg_hi_dict)

        return hi_dict, deg_hi_dict, nt_summary
