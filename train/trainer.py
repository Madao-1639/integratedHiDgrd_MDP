import torch
from torch import nn
from torch.optim import Adam

import numpy as np
import pandas as pd
# from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from utils.data import select_loader
from model import BaseRTF, BaseTW, SC_DNN, Integrated_DNN_LSTM, MSRTF#, MSTW
from loss import FocalLoss,MVFLoss,MONLoss,CONLoss
from utils.logger import Logger
from utils.utils import test4norm, plot_hi



from abc import ABC, abstractmethod
class BaseTrainer(ABC):
    """Base class for trainers."""
    def __init__(self,args,train_data: "pd.DataFrame",val_data: "pd.DataFrame" =None, **logger_kwargs) -> None:
        self.args = args
        if args.logger:
            self.logger = Logger(args,**logger_kwargs)
        else:
            self.logger = None
        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f'Training on {self.device}')
        self.get_loader(train_data,val_data)
        self.get_model()
        self.get_optimizer()

        self.get_loss_wa_coef()

    def get_loader(self,train_data: "pd.DataFrame", val_data: "pd.DataFrame") -> None:
        self.train_loader = select_loader(train_data,True,self.args)
        self.ls_dict = self.train_loader.dataset.ls_dict

        if val_data is not None:
            self.val_loader = select_loader(val_data,False,self.args)
        else:
            self.val_loader = None

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

            if self.logger:
                # Sample/Select UUTs to plot HI
                data_record_UUTs = record_HI_data['UUT'].unique()
                args_record_UUTs = self.args.record_UUTs
                if args_record_UUTs and all((data_record_UUTs==UUT).any() for UUT in args_record_UUTs):
                    self.record_UUTs = args_record_UUTs
                else:
                    record_num_UUTs = min(data_record_UUTs.size,self.args.record_num_UUTs)
                    self.record_UUTs = np.random.choice(data_record_UUTs,size=record_num_UUTs,replace=False)
        else:
            self.record_HI_loader = None

    @abstractmethod
    def get_model(self) -> None:
        raise NotImplementedError

    def get_optimizer(self) -> None:
        no_decay_pg, decay_pg = [], []
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm1d):
                no_decay_pg.extend(layer.parameters())
            else:
                for name, param in layer.named_parameters(recurse=False):
                    if 'weight' in name:
                        decay_pg.append(param)
                    else:
                        no_decay_pg.append(param)
        optimizer = Adam([{'params':no_decay_pg}], lr=self.args.lr)
        optimizer.add_param_group({'params': decay_pg, 'weight_decay': self.args.weight_decay})
        self.optimizer = optimizer

    def get_loss_wa_coef(self) -> None:
        self.train_UUT_dict = {UUT:i for i,UUT in enumerate(self.ls_dict.index)}
        pass

    def _UUT2idx(self,UUT):
        if hasattr(UUT,'__getitem__'): # The passed UUT is a sequence
            return [self.train_UUT_dict[_] for _ in UUT]
        else:
            return self.train_UUT_dict[UUT]

    def train(self):
        for epoch in range(1,self.args.num_epoch+1):
            # Train Stage
            self.train_per_epoch(epoch)
            if self.logger:
                # Val Stage (log cls metrics)
                if self.val_loader is not None:
                    self.val_per_epoch(epoch)
                # Record HI
                if self.record_HI_loader is not None:
                    self.record_per_epoch(epoch)
                # Log results
                self.logger.save_metrics(epoch)
                self.logger.save_checkpoint(self.model, epoch)

    @abstractmethod
    def train_per_epoch(self,epoch: "int"):
        raise NotImplementedError

    def val_per_epoch(self,epoch: "int") -> dict:
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
        if self.logger:
            for metric_name,metric in metric_result.items():
                self.logger.writer.add_scalar(f'Metric/val_{metric_name}',metric,epoch)
        return metric_result

    def record_per_epoch(self,epoch: "int") -> dict:
        self.model.eval()

        # Record HI
        hi_dict = {}
        for UUT,t,X,y_true in self.record_HI_loader:
            if UUT in self.record_UUTs:
                X = X.to(self.device)
                hi = self.model(X).detach().numpy()
                hi_dict[UUT] = hi

        # Plot selected HI
        if self.logger and (epoch % self.args.record_freq == 0):
            fig = plot_hi(hi_dict,self.record_UUTs)
            self.logger.writer.add_figure(f'HI/{self.args.record_HI}',fig,epoch)

        return hi_dict



class BaseRTFTrainer(BaseTrainer):
    '''BaseRTF Model Trainer'''

    def get_model(self):
        self.model = BaseRTF(self.args, self.ls_dict.index)
        if self.args.load_model_fp:
            self.model.load_state_dict(torch.load(self.args.load_model_fp, weights_only=False),strict=False)
        self.model.to(self.device)
        # example_input = torch.randn((self.args.input_size,20))
        # self.logger.writer.add_graph(self.model,example_input)

    def train_per_epoch(self, epoch):
        # Switch to train mode
        self.model.train()

        for i, (UUT, t, X, y_true) in enumerate(self.train_loader):
            X = X.to(self.device)
            y_true = y_true.to(self.device)

            hi, p = self.model(X)

            # Compute loss
            loss = self.compute_loss(hi, p, y_true, UUT)

            # Get the item for backward
            total_loss = loss['total_loss']

            # Compute gradient and do Adam step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Constrain sigma_square to be big enough
            self.model.sigma_square.data.clamp_(0.1)

            # Logger record loss
            if self.logger:
                for k,v in loss.items():
                    self.logger.record_scalars('Loss/train', k, v)

            # Monitor training progress
            if (i+1) % self.args.print_freq == 0:
                print(f'Train: Epoch {epoch} batch {i+1} Loss {total_loss.item():.6f}')

    def record_per_epoch(self,epoch):
        self.model.eval()

        # Record HI
        hi_dict = {}
        for UUT,t,X,y_true in self.record_HI_loader:
            X = X.to(self.device)
            hi,p = self.model(X)
            hi_dict[UUT] = hi.detach().numpy()
        if self.logger:
            # Log parameters
            self.logger.writer.add_histogram('theta/train',self.model.theta_train,epoch)
            self.logger.writer.add_scalar('sigma_square',self.model.sigma_square,epoch)

            # Test for normality
            nt_summary = test4norm(hi_dict,sig_list=(0.01,0.05,0.10))
            for test_name, test_result in nt_summary.items():
                self.logger.writer.add_scalar(f'NomalTest/{self.args.record_HI}_{test_name}',test_result,epoch)

            # Plot selected HI
            if epoch % self.args.record_freq == 0:
                fig = plot_hi(hi_dict,self.record_UUTs)
                self.logger.writer.add_figure(f'HI/{self.args.record_HI}',fig,epoch)

        return hi_dict

    def compute_loss(self, hi, p, y_true, UUT):
        cls_loss = self.args.cls_loss_weight * FocalLoss(p, y_true, alpha = self.args.FocalLoss_alpha, gamma = self.args.FocalLoss_gamma, reduction='mean')
        mfe_loss = self.args.mfe_loss_weight * self.model.mfe_loss(hi, UUT, reduction = 'mean')

        total_loss = cls_loss + mfe_loss
        loss = {
            'cls_loss': cls_loss,
            'mfe_loss': mfe_loss,
            'total_loss': total_loss
        }
        return loss



class BaseTWTrainer(BaseRTFTrainer):
    '''BaseTW Model Trainer'''

    def get_model(self):
        self.model = BaseTW(self.args, self.ls_dict.index)
        if self.args.load_model_fp:
            self.model.load_state_dict(torch.load(self.args.load_model_fp, weights_only=False),strict=False)
        self.model.to(self.device)
        # example_input = torch.randn((self.args.window_width,self.args.input_size,))
        # self.logger.writer.add_graph(self.model,example_input)

    def get_loss_wa_coef(self):
        # Coefficients of Weighted-Average(WA) of loss for each UUT.
        super().get_loss_wa_coef()
        self.cls_loss_wa_coef = torch.FloatTensor(1/(self.ls_dict.values-self.args.window_width+1))
        self.mfe_loss_wa_coef = torch.FloatTensor(1/(self.ls_dict.values-self.args.window_width))

    def train_per_epoch(self, epoch):
        self.model.train()

        for i, (start, end, UUT, t, (X_pre, X_cur), (y_pre, y_cur)) in enumerate(self.train_loader):
            X_pre = X_pre.to(self.device)
            X_cur = X_cur.to(self.device)
            y_pre = y_pre.to(self.device)
            y_cur = y_cur.to(self.device)

            hi_pre, p_pre = self.model(X_pre)
            hi_cur, p_cur = self.model(X_cur)

            loss = self.compute_loss(start, end, hi_pre, hi_cur, p_pre, p_cur, y_pre, y_cur, UUT)

            total_loss = loss['total_loss']

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if self.logger:
                for k,v in loss.items():
                    self.logger.record_scalars('Loss/train', k, v)

            if (i+1) % self.args.print_freq == 0:
                print(f'Train: Epoch {epoch} batch {i+1} Loss {total_loss.item():.6f}')

    def compute_loss(self, start, end, hi_pre, hi_cur, p_pre, p_cur, y_pre, y_cur, UUT):
        indice = self._UUT2idx(UUT)
        cls_loss_wa_coef = self.cls_loss_wa_coef[indice]
        mfe_loss_wa_coef = self.mfe_loss_wa_coef[indice]

        cls_loss_start = cls_loss_wa_coef[start]@FocalLoss(p_pre[start], y_pre[start], alpha = self.args.FocalLoss_alpha, gamma = self.args.FocalLoss_gamma, reduction='none')
        cls_loss_cur= cls_loss_wa_coef@FocalLoss(p_cur, y_cur, alpha = self.args.FocalLoss_alpha, gamma = self.args.FocalLoss_gamma, reduction='none')
        cls_loss = self.args.cls_loss_weight * (cls_loss_start + cls_loss_cur)

        mfe_loss = self.args.mfe_loss_weight * (mfe_loss_wa_coef@self.model.mfe_loss(hi_pre, hi_cur, UUT, reduction = 'none'))

        total_loss = cls_loss + mfe_loss
        loss = {
            'cls_loss': cls_loss,
            'mfe_loss': mfe_loss,
            'total_loss': total_loss
        }
        return loss



class SCTrainer(BaseTrainer):
    '''SC Model Trainer'''

    def get_model(self):
        self.model = SC_DNN(self.args)
        if self.args.load_model_fp:
            self.model.load_state_dict(torch.load(self.args.load_model_fp, weights_only=False),strict=False)
        self.model.to(self.device)
        # example_input = torch.randn((self.args.input_size,))
        # self.logger.writer.add_graph(self.model,example_input)

    def get_loss_wa_coef(self):
        super().get_loss_wa_coef()
        self.mon_loss_wa_coef = torch.FloatTensor(1/(self.ls_dict.values-1))
        self.con_loss_wa_coef = torch.FloatTensor(1/(self.ls_dict.values-2))

    def train_per_epoch(self, epoch):
        # Switch to train mode
        self.model.train()

        all_y = []
        all_hi = []
        for i, (start, end, UUT, t, (X_ppre, X_pre, X_cur), (y_ppre, y_pre, y_cur)) in enumerate(self.train_loader):
            X_ppre = X_ppre.to(self.device)
            X_pre = X_pre.to(self.device)
            X_cur = X_cur.to(self.device)
            # y_true = y_true.to(self.device)

            hi_ppre = self.model(X_ppre)
            hi_pre = self.model(X_pre)
            hi_cur = self.model(X_cur)

            loss = self.compute_loss(start, end, hi_ppre, hi_pre, hi_cur, UUT) # unsupervised Learning

            total_loss = loss['total_loss']

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if self.logger:
                for k,v in loss.items():
                    self.logger.record_scalars('Loss/train', k, v)

            if (i+1) % self.args.print_freq == 0:
                print(f'Train: Epoch {epoch} batch {i+1} Loss {total_loss.item():.6f}')
            
            all_y.extend([y_ppre[start],y_pre[start],y_cur])
            all_hi.extend([hi_ppre[start].detach(),hi_pre[start].detach(),hi_cur.detach()])
        all_y = torch.concat(all_y)
        all_hi = torch.concat(all_hi)
        self.model.fit(all_hi,all_y)

    def compute_loss(self, start, end, hi_ppre, hi_pre, hi_cur, UUT):
        indice = self._UUT2idx(UUT)
        mon_loss_wa_coef = self.mon_loss_wa_coef[indice]
        con_loss_wa_coef = self.con_loss_wa_coef[indice]

        mvf_loss = self.args.mvf_loss_weight * MVFLoss(hi_cur[end], self.args.MVFLoss_m, reduction="sum")

        mon_loss_start = mon_loss_wa_coef[start]@MONLoss(hi_ppre[start],hi_pre[start], c=self.args.MONLoss_c, reduction="none")
        mon_loss_cur = mon_loss_wa_coef@MONLoss(hi_pre,hi_cur, c=self.args.MONLoss_c, reduction="none")
        mon_loss = self.args.mon_loss_weight * (mon_loss_start + mon_loss_cur)
        con_loss = self.args.con_loss_weight * (con_loss_wa_coef@CONLoss(hi_ppre,hi_pre,hi_cur, c=self.args.CONLoss_c, reduction="none"))

        total_loss = mvf_loss + mon_loss + con_loss
        loss = {
            'mvf_loss': mvf_loss,
            'con_loss': con_loss,
            'mon_loss': mon_loss,
            'total_loss': total_loss
        }
        return loss



class IntegratedTrainer(BaseTrainer):
    '''Integrated Model Trainer'''

    def get_model(self):
        self.model = Integrated_DNN_LSTM(self.args, self.ls_dict.index)
        if self.args.load_model_fp:
            self.model.load_state_dict(torch.load(self.args.load_model_fp, weights_only=False),strict=False)
        self.model.to(self.device)
        # example_input = torch.randn((self.args.input_size,20))
        # self.logger.writer.add_graph(self.model,example_input)

    def train_per_epoch(self, epoch):
        self.model.train()

        all_y = []
        all_hi = []
        for i, (UUT, t, X, y_true) in enumerate(self.train_loader):
            t = t.to(self.device)
            X = X.to(self.device)
            y_true = y_true.to(self.device)

            hi = self.model(X)

            loss = self.compute_loss(hi, t, UUT)

            total_loss = loss['total_loss']

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if self.logger:
                for k,v in loss.items():
                    self.logger.record_scalars('Loss/train', k, v)

            if (i+1) % self.args.print_freq == 0:
                print(f'Train: Epoch {epoch} batch {i+1} Loss {total_loss.item():.6f}')
            
            all_y.append(y_true)
            all_hi.append(hi.detach())
        all_y = torch.concat(all_y)
        all_hi = torch.concat(all_hi)
        self.model.fit(all_hi,all_y)

    def compute_loss(self, hi, t, UUT):
        mfe_loss = self.args.mfe_loss_weight * self.model.mfe_loss(hi, t, UUT, reduction = 'mean')
        mvf_loss = self.args.mvf_loss_weight * MVFLoss(hi[-1], self.args.MVFLoss_m, reduction="none")
        mon_loss = self.args.mon_loss_weight * MONLoss(hi, c=self.args.MONLoss_c, reduction="mean")
        con_loss = self.args.mon_loss_weight * CONLoss(hi, c=self.args.CONLoss_c, reduction="mean")

        total_loss = mfe_loss + mvf_loss + mon_loss + con_loss
        loss = {
            'mfe_loss': mfe_loss,
            'mvf_loss': mvf_loss,
            'con_loss': con_loss,
            'mon_loss': mon_loss,
            'total_loss': total_loss
        }
        return loss



class MSRTFTrainer(BaseRTFTrainer):
    '''MSRTF Model Trainer'''

    def get_model(self):
        self.model = MSRTF(args=self.args, train_UUTs=self.ls_dict.index)
        if self.args.load_model_fp:
            self.model.load_state_dict(torch.load(self.args.load_model_fp, weights_only=False),strict=False)
        self.model.to(self.device)
        # example_input = torch.randn((self.args.input_size,20))
        # self.logger.writer.add_graph(self.model,example_input)

    def record_per_epoch(self,epoch):
        self.model.eval()

        # Record HI
        hi_dict = {}
        deg_hi_dict = {}
        for UUT,t,X,y_true in self.record_HI_loader:
            X = X.to(self.device)
            hi,p = self.model(X)
            hi_dict[UUT] = hi.detach().numpy()
            deg_hi_dict[UUT] = self.model.transform_deg_hi(hi).detach().numpy()

        if self.logger:
            # Log parameters
            self.logger.writer.add_histogram('theta/train',self.model.theta_train,epoch)
            self.logger.writer.add_scalar('sigma_square',self.model.sigma_square,epoch)
            for param, value in self.model.hi_transformer.named_parameters(recurse=False):
                self.logger.writer.add_scalar(f'LLT/{param}', value, epoch)
            if self.args.MS_flex_type is not None:
                for param, value in self.model.get_flex_coef.named_parameters(recurse=False):
                    self.logger.writer.add_scalar(f'{self.args.MS_flex_type}/{param}',value,epoch)

            # Test for normality
            nt_summary = test4norm(deg_hi_dict,sig_list=(0.01,0.05,0.10))
            for test_name, test_result in nt_summary.items():
                self.logger.writer.add_scalar(f'NomalTest/{self.args.record_HI}_{test_name}',test_result,epoch)

            # Plot selected HI
            if epoch % self.args.record_freq == 0:
                hi_fig = plot_hi(hi_dict,self.record_UUTs)
                self.logger.writer.add_figure(f'HI/{self.args.record_HI}',hi_fig,epoch)
                deg_hi_fig = plot_hi(deg_hi_dict,self.record_UUTs)
                self.logger.writer.add_figure(f'deg_HI/{self.args.record_HI}',deg_hi_fig,epoch)

        return hi_dict, deg_hi_dict