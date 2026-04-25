import os
import pickle

import torch
from torch import nn

import numpy as np
import pandas as pd
# from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

from torch.utils.data import DataLoader
from utils.data import DATASET_REGISTRY
from model import MODEL_REGISTRY
from loss import OPTIMIZER_REGISTRY
from utils.registry import TRAINER_REGISTRY
from loss import FocalLoss, MFELoss, MVFLoss, MONLoss, CONLoss
from utils.logger import Logger
from utils.utils import test4norm, plot_hi



from abc import ABC, abstractmethod
class BaseTrainer(ABC):
    """Base class for trainers."""
    def __init__(self, args,
                train_data: pd.DataFrame | None = None, val_data: pd.DataFrame | None = None,
                train_loader: DataLoader | None = None, val_loader: DataLoader | None = None,
                 **logger_kwargs) -> None:
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
        self.get_loader(train_data, val_data, train_loader, val_loader)
        self.get_model()
        self.get_optimizer()
        self.get_loss_wa_coef()

    def get_loader(self,
                train_data: pd.DataFrame | None = None, val_data: pd.DataFrame | None = None,
                train_loader: DataLoader | None = None, val_loader: DataLoader | None = None) -> None:
        if train_data is not None and not train_loader:
            train_dataset = DATASET_REGISTRY[self.args.data_type](train_data, train=True, args=self.args)
            self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        elif train_loader:
            self.train_loader = train_loader
        else:
            raise ValueError('Please provide either train_data or train_loader.')
        self.ls_dict = self.train_loader.dataset.ls_dict

        if val_data is not None and not val_loader:
            val_dataset = DATASET_REGISTRY[self.args.data_type](val_data, train=False, args=self.args)
            self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
        elif val_loader:
            self.val_loader = val_loader
        else:
            self.val_loader = None

    def get_model(self) -> None:
        if self.args.load_model_fp:
            self.model = torch.load(self.args.load_model_fp, map_location=self.device, weights_only=False)
        else:
            self.model = MODEL_REGISTRY[self.args.model_type](self.args)

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
        optimizer = OPTIMIZER_REGISTRY[self.args.optimizer]([{'params':no_decay_pg}], lr=self.args.lr)
        optimizer.add_param_group({'params': decay_pg, 'weight_decay': self.args.weight_decay})
        self.optimizer = optimizer

    def get_loss_wa_coef(self) -> None:
        self.train_UUT_dict = {UUT:i for i,UUT in enumerate(self.ls_dict.index)}
        pass

    def _UUT2idx(self, UUT):
        if isinstance(UUT, (torch.Tensor, np.ndarray, pd.Series, list, tuple, set, dict)):
            return [self.train_UUT_dict[_] for _ in UUT]
        else:
            return self.train_UUT_dict[UUT]

    def move_batch_to_device(self, batch: dict) -> None:
        # Move all tensor data in the batch to the model's device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
    
    @abstractmethod
    def model_forward(self, batch: dict) -> dict:
        '''
        Execute a single forward pass through the model.

        This abstract method defines the core model forward logic. 
        Implementations must include two key components:
        1. Forward pass: Compute model outputs from input data.
        2. Loss computation: Calculate the loss value based on predictions and ground truth.

        Args:
            batch(dict): The input data batch for training.

        Returns:
            dict: A dictionary containing model outputs and other relevant information.
        '''
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, output: dict, batch: dict) -> dict:
        '''
        Compute the loss for a given batch of data.

        Args:
            output (dict): The model's output for the given batch.
            batch (dict): The input data batch.

        Returns:
            A dictionary containing the computed loss.
        '''
        raise NotImplementedError

    def model_predict(self, batch: dict) -> dict:
        return self.model_forward(batch)
    
    def compute_metrics(self, output: dict, batch: dict) -> dict:
        precision, recall, f1, _ = precision_recall_fscore_support(
            batch['Y'], output['Y'], 
            average = 'binary', zero_division = 0
        )
        return {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
        }
    def optimize(self, total_loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def train(self):
        for epoch in range(1, self.args.num_epoch + 1):
            # Train Stage
            self.train_per_epoch(epoch)
            if self.logger:
                # Val Stage (log cls metrics)
                if self.val_loader:
                    self.val_per_epoch(epoch)
                # Log results
                self.logger.save_metrics(epoch)
                self.logger.save_checkpoint(self.model, epoch)

    def train_per_epoch(self, epoch: int):
        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            self.before_train_step(batch_idx, batch)
            loss = self.train_step(batch)
            self.after_train_step(epoch, batch_idx, loss)

    def before_train_step(self, batch_idx: int, batch: dict) -> None:
        """
        Hook before each training step.

        Args:
            batch_idx (int): The index of the current batch.
            batch (dict): A dictionary containing the batch data, where values may be torch.Tensors.
        """
        self.move_batch_to_device(batch)
    
    def train_step(self, batch: dict) -> dict:
        """Execute a single training step.

        This method defines the core training logic for one batch. 
        Implementations must include three key components:
        1. Forward pass: Compute model outputs from input data.
        2. Loss computation: Calculate the loss value based on predictions and ground truth.
        3. Gradient update: Perform backpropagation and update model parameters.

        Args:
            batch(dict): The input data batch for training.

        Returns:
            A dictionary containing the computed loss values.
        """
        output = self.model_forward(batch)
        loss = self.compute_loss(output, batch)
        self.optimize(loss['total_loss'])
    
        return loss

    def after_train_step(self, epoch: int, batch_idx: int, loss: dict) -> None:
        # Record loss
        if self.logger:
            for k, v in loss.items():
                self.logger.record_scalars('Loss/train', k, v)

        # Monitor training progress
        if (batch_idx + 1) % self.args.print_freq == 0:
            print(f'Train: Epoch {epoch} batch {batch_idx + 1} Loss {loss['total_loss'].item():.6f}')

    def val_per_epoch(self, epoch: int) -> dict:
        self.model.eval()
        all_metrics = {}
        for batch_idx, batch in enumerate(self.val_loader):
            self.before_val_step(batch_idx, batch)
            metrics = self.val_step(batch)
            self.after_val_step(epoch, batch_idx, metrics, all_metrics)
        for metric_name, metric in all_metrics.items():
            self.logger.writer.add_scalar(f'Metric/{metric_name}', sum(metric)/len(metric), epoch)

    def before_val_step(self, batch_idx: int, batch: dict) -> None:
        self.move_batch_to_device(batch)

    def val_step(self, batch: dict) -> dict:
        output = self.model_predict(batch)
        metrics = self.compute_metrics(output, batch)
        return metrics
    
    def after_val_step(self, epoch: int, batch_idx: int, metrics: dict, all_metrics: dict) -> None:
        # Record metrics
        for metric_name, metric in metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = [metric]


@TRAINER_REGISTRY('Base')
class BaseRTFTrainer(BaseTrainer):
    '''BaseRTF Model Trainer'''

    def get_model(self):
        super().get_model()
        self.mfe_loss = MFELoss(n_UUT=len(self.ls_dict))
    
    def before_train_step(self, batch_idx, batch):
        super().before_train_step(batch_idx, batch)
        batch['lengths'] = torch.as_tensor([t.shape[0] for t in batch['t']])

    def model_forward(self, batch):
        hi, mask = self.model(batch['X'], batch['lengths'])
        return {
            'hi': hi,
            'mask': mask
        }
    
    def compute_loss(self, output, batch):
        hi, mask = output['hi'], output['mask']
        indice = torch.tensor(self._UUT2idx(batch['UUT']))
        cls_loss = FocalLoss(hi, batch['Y'], alpha = self.args.FocalLoss_alpha, gamma = self.args.FocalLoss_gamma, reduction = 'none')
        cls_loss = self.args.cls_loss_weight * cls_loss.masked_select(mask).mean()
        mfe_loss = self.mfe_loss(hi, indice, reduction = 'none')
        mfe_loss = self.args.mfe_loss_weight * mfe_loss.masked_select(mask).mean()

        total_loss = cls_loss + mfe_loss
        return {
            'cls_loss': cls_loss,
            'mfe_loss': mfe_loss,
            'total_loss': total_loss
        }
    
    def after_train_step(self, epoch, batch_idx, loss):
        self.constrain_parameters()
        super().after_train_step(epoch, batch_idx, loss)

    def model_predict(self, batch):
        return {
            'Y': self.model.predict(batch['X'], batch['lengths']),
        }

    def record_per_epoch(self, epoch):
        self.model.eval()

        # Record HI
        hi_dict = {}
        for batch_UUT, batch_t, batch_X, batch_Y in self.record_HI_loader:
            lengths = torch.as_tensor([t.size(0) for t in batch_t])
            batch_X = batch_X.to(self.device)
            batch_Y = batch_Y.to(self.device)
            hi, mask = self.model(batch_X, lengths)
            for i, UUT in enumerate(batch_UUT):
                hi_dict[UUT] = hi[i, :lengths[i]].detach().numpy()

        if self.logger:
            # Log parameters
            self.logger.writer.add_histogram('theta/train',self.mfe_loss.theta_train, epoch)
            self.logger.writer.add_scalar('sigma_square',self.mfe_loss.sigma_square, epoch)

            # Test for normality
            nt_summary = test4norm(hi_dict)
            for test_name, test_result in nt_summary.items():
                self.logger.writer.add_scalar(f'NomalTest/{self.args.record_HI}_{test_name}',test_result,epoch)

            # Plot selected HI
            if epoch % self.args.record_freq == 0:
                fig = plot_hi(hi_dict, self.record_UUTs)
                self.logger.writer.add_figure(f'HI/{self.args.record_HI}', fig, epoch)

        return hi_dict

    def constrain_parameters(self):
        # Constrain sigma_square to be big enough
        self.mfe_loss.sigma_square.data.clamp_(min = 0.1)


@TRAINER_REGISTRY('SC')
class SCTrainer(BaseTrainer):
    '''SC Model Trainer'''

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
        indices = self._UUT2idx(UUT)
        mon_loss_wa_coef = self.mon_loss_wa_coef[indices]
        con_loss_wa_coef = self.con_loss_wa_coef[indices]

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


@TRAINER_REGISTRY('Integrated')
class IntegratedTrainer(BaseTrainer):
    '''Integrated Model Trainer'''

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


@TRAINER_REGISTRY('MS')
class MSRTFTrainer(BaseRTFTrainer):
    '''MSRTF Model Trainer'''
    def model_forward(self, batch):
        logits, hi, cls_mask, mfe_mask = self.model(batch['X'], batch['lengths'])
        return {
            'logits': logits,
            'hi': hi,
            'cls_mask': cls_mask,
            'mfe_mask': mfe_mask,
        }
    
    def compute_loss(self, output, batch):
        logits, hi, cls_mask, mfe_mask = output['logits'], output['hi'], output['cls_mask'], output['mfe_mask']
        indice = torch.tensor(self._UUT2idx(batch['UUT']))
        cls_loss = FocalLoss(logits, batch['Y'], alpha = self.args.FocalLoss_alpha, gamma = self.args.FocalLoss_gamma, reduction = 'none')
        cls_loss = self.args.cls_loss_weight * cls_loss.masked_select(cls_mask).mean()
        mfe_loss = self.mfe_loss(hi, indice, reduction = 'none')
        mfe_loss = self.args.mfe_loss_weight * mfe_loss.masked_select(mfe_mask).mean()
        total_loss = cls_loss + mfe_loss
        return {
            'cls_loss': cls_loss,
            'mfe_loss': mfe_loss,
            'total_loss': total_loss
        }
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
            nt_summary = test4norm(deg_hi_dict)
            for test_name, test_result in nt_summary.items():
                self.logger.writer.add_scalar(f'NomalTest/{self.args.record_HI}_{test_name}',test_result,epoch)

            # Plot selected HI
            if epoch % self.args.record_freq == 0:
                hi_fig = plot_hi(hi_dict,self.record_UUTs)
                self.logger.writer.add_figure(f'HI/{self.args.record_HI}',hi_fig,epoch)
                deg_hi_fig = plot_hi(deg_hi_dict,self.record_UUTs)
                self.logger.writer.add_figure(f'deg_HI/{self.args.record_HI}',deg_hi_fig,epoch)

        return hi_dict, deg_hi_dict

    # def constrain_parameters(self):
    #     super().constrain_parameters()
    #     self.model.hi_transformer.c1.data.clamp_(max=11)
    #     self.model.hi_transformer.c2.data.clamp_(max=0.1)
