import os
import pickle
import torch

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from utils.data import select_loader
from model import BaseRTF, BaseTW, SC_DNN, Integrated_DNN_LSTM, MSRTF#, MSTW
from utils.utils import test4norm, plot_hi



from abc import ABC, abstractmethod
class BaseEvaluator(ABC):
    """Base class for evaluators."""
    def __init__(self, args,
                train_data: pd.DataFrame = None, val_data: pd.DataFrame = None,
                 **logger_kwargs) -> None:
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

    @abstractmethod
    def get_model(self) -> None:
        """
        Loads model if `load_model_fp` is provided, 
        and moves the model to the configured device.
        This method should be called in the subclass after defining `self.model`.
        """
        if self.args.load_model_fp:
            # Parse file paths
            network_fp = self.args.load_model_fp
            if network_fp.endswith('.pth'):
                classifier_fp = network_fp[:-4]+'.pkl'
            else:
                classifier_fp = network_fp+'.pkl'
                network_fp = network_fp+'.pth'
            # Load network
            self.model.load_state_dict(torch.load(network_fp, weights_only=False),strict=False)
            # Load classifier if exists (for Integrated & SC models)
            if os.path.exists(classifier_fp):
                with open(classifier_fp,'rb') as classifier_f:
                    self.model.cls_model = pickle.load(classifier_f)
        self.model.to(self.device)

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

    def get_model(self):
        self.model = BaseRTF(self.args, train_UUTs=[])
        del self.model.theta_train
        # example_input = torch.randn((self.args.input_size,20))
        # self.logger.writer.add_graph(self.model,example_input)
        super().get_model()

    def record(self):
        self.model.eval()

        # Record HI
        hi_dict = {}
        for UUT,t,X,y_true in self.record_HI_loader:
            X = X.to(self.device)
            hi,p = self.model(X)
            hi_dict[UUT] = hi.detach().numpy()

        # Test for normality
        nt_summary = test4norm(hi_dict,sig_list=(0.01,0.05,0.10))

        return hi_dict, nt_summary



class BaseTWEvaluator(BaseRTFEvaluator):
    '''BaseTW Model Evaluator'''

    def get_model(self):
        self.model = BaseTW(self.args, train_UUTs=[])
        # example_input = torch.randn((self.args.window_width,self.args.input_size,))
        # self.logger.writer.add_graph(self.model,example_input)
        del self.model.theta_train
        super().get_model()



class SCEvaluator(BaseEvaluator):
    '''SC Model Evaluator'''

    def get_model(self):
        self.model = SC_DNN(self.args)
        # example_input = torch.randn((self.args.input_size,))
        # self.logger.writer.add_graph(self.model,example_input)
        super().get_model()



class IntegratedEvaluator(BaseEvaluator):
    '''Integrated Model Evaluator'''

    def get_model(self):
        self.model = Integrated_DNN_LSTM(self.args,train_UUTs=[])
        # example_input = torch.randn((self.args.input_size,20))
        # self.logger.writer.add_graph(self.model,example_input)
        del self.model.Gamma_train
        super().get_model()



class MSRTFEvaluator(BaseRTFEvaluator):
    '''MSRTF Model Evaluator'''

    def get_model(self):
        self.model = MSRTF(args=self.args, train_UUTs=[])
        # example_input = torch.randn((self.args.input_size,20))
        # self.logger.writer.add_graph(self.model,example_input)
        del self.model.theta_train
        super(BaseRTFEvaluator,self).get_model()

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
        nt_summary = test4norm(deg_hi_dict,sig_list=(0.01,0.05,0.10))

        return hi_dict, deg_hi_dict, nt_summary
