import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import torch
from torch.utils.data import DataLoader
from utils.data import DATASET_REGISTRY
from utils.registry import EVALUATOR_REGISTRY
from utils.utils import test4norm


# from abc import ABC, abstractmethod
class BaseEvaluator:
    '''Base class for evaluators.'''
    def __init__(self, args,
                 train_data: pd.DataFrame = None, val_data: pd.DataFrame = None) -> None:
        self.args = args
        if args.use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f'Evaluating on {self.device}')
        self.get_loader(val_data=val_data, train_data=train_data)
        self.get_model()

    def get_loader(self, val_data: pd.DataFrame, train_data: pd.DataFrame | None = None) -> None:
        val_dataset = DATASET_REGISTRY[self.args.data_type](val_data, train = False, args = self.args)
        self.val_loader = DataLoader(val_dataset, batch_size = self.args.batch_size, shuffle = False, collate_fn = val_dataset.collate_fn)

        if self.args.record_HI:
            if self.args.record_HI == 'train':
                record_HI_data = train_data
            elif self.args.record_HI == 'val':
                record_HI_data = val_data
            else:
                record_HI_data = pd.concat([train_data, val_data])
            # Try to select valid given UUTs
            data_record_UUTs = record_HI_data['UUT'].unique()
            args_record_UUTs = self.args.record_UUTs
            record_UUTs = list(set(data_record_UUTs) & set(args_record_UUTs))
            if len(record_UUTs) > 0:
                record_HI_data = record_HI_data[record_HI_data['UUT'].isin(record_UUTs)]
            else:
                # No valid UUTs, try to draw given number of UUTs
                record_num_UUTs = min(data_record_UUTs.size, self.args.record_num_UUTs)
                if record_num_UUTs > 0:
                    record_UUTs = np.random.choice(data_record_UUTs, size = record_num_UUTs, replace = False)
                    record_HI_data = record_HI_data[record_HI_data['UUT'].isin(record_UUTs)]
            record_HI_dataset = DATASET_REGISTRY['RTF'](record_HI_data, train = False, args = self.args) # Restrict record_HI_dataset to RTF Dataset
            self.record_HI_loader = DataLoader(record_HI_dataset, batch_size = self.args.batch_size, shuffle = False, collate_fn = record_HI_dataset.collate_fn)
        else:
            self.record_HI_loader = None

    def get_model(self) -> None:
        self.model = torch.load(self.args.load_model_fp, map_location = self.device, weights_only = False)

    def move_batch_to_device(self, batch: dict) -> None:
        # move all tensor data in the batch to the model's device
        def _move(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            elif isinstance(x, list):
                for i, x_i in enumerate(x):
                    x[i] = _move(x_i)
                return x
            elif isinstance(x, tuple):
                return tuple(_move(x_i) for x_i in x)
            else:
                return x

        for k, v in batch.items():
            batch[k] = _move(v)

    def model_predict(self, batch: dict) -> dict:
       hi = self.model(batch['X']).detach()
       Y_pred = self.model.cls_model.predict(hi.unsqueeze(-1))
       return {
           'Y': Y_pred,
       } 

    def compute_metrics(self, meta) -> dict:
        Y_true, Y_pred = meta['Y_true'], meta['Y_pred']
        precision, recall, f1, _ = precision_recall_fscore_support(
            Y_true, Y_pred, average='binary', zero_division=0
        )
        return {
            'F1': f1,
            'precision': precision,
            'recall': recall,
        }

    def evaluate(self) -> dict:
        '''
        Validate the model. This method iterates over the validation data loader and calls the appropriate hooks for each validation stage.
        '''
        meta = self.on_val_start()
        for batch_idx, batch in enumerate(self.val_loader):
            self.before_val_step(batch_idx, batch, meta)
            result = self.val_step(batch)
            self.after_val_step(batch_idx, batch, result, meta)
        return self.on_val_end(meta)

    def on_val_start(self) -> None:
        self.model.eval()
        return {
            'Y_true': [],
            'Y_pred': [],
        }
    
    def before_val_step(self, batch_idx: int, batch: dict, meta: dict) -> None:
        self.move_batch_to_device(batch)

    def val_step(self, batch: dict) -> dict:
        return {
            'output': self.model_predict(batch),
        }
    
    def after_val_step(self, batch_idx: int, batch: dict, result: dict, meta: dict) -> None:
        meta['Y_true'].append(batch['Y'])
        meta['Y_pred'].append(result['output']['Y'])

    def on_val_end(self, meta: dict) -> None:
        meta['Y_true'] = np.concatenate(meta['Y_true'])
        meta['Y_pred'] = np.concatenate(meta['Y_pred'])
        return self.compute_metrics(meta)

    def model_forward(self, batch: dict) -> dict:
        return {
            'hi': self.model(batch['X']),
        }

    def record(self) -> dict:
        '''
        Record health indicators (HI). This method iterates over the record_HI_loader and calls the appropriate hooks for each record stage.
        '''
        meta = self.on_record_start()
        for batch_idx, batch in enumerate(self.record_HI_loader):
            self.before_record_step(batch_idx, batch, meta)
            result = self.record_step(batch)
            self.after_record_step(batch_idx, batch, result, meta)
        return self.on_record_end(meta)

    def on_record_start(self) -> None:
        self.model.eval()
        return {
            'hi_dict': {},
        }

    def before_record_step(self, batch_idx: int, batch: dict, meta: dict) -> None:
        self.move_batch_to_device(batch)

    def record_step(self, batch: dict) -> dict:
        return {
            'output': self.model_forward(batch),
        }
    
    def after_record_step(self, batch_idx: int, batch: dict, result: dict, meta: dict) -> None:
        batch_hi = result['output']['hi'].detach().numpy()
        batch_UUT, lengths = batch['UUT'], batch['lengths']
        for UUT, hi, length in zip(batch_UUT, batch_hi, lengths):
            meta['hi_dict'][UUT] = hi[:length]

    def on_record_end(self, meta: dict) -> None:
        return meta

EVALUATOR_REGISTRY.register('SC', BaseEvaluator)
EVALUATOR_REGISTRY.register('Integrated', BaseEvaluator)


@EVALUATOR_REGISTRY('Base')
class BaseRTFEvaluator(BaseEvaluator):
    '''BaseRTF Model Evaluator'''

    def model_predict(self, batch):
        Y_pred, mask = self.model.predict(batch['X'], batch['lengths'])
        return {
            'Y': Y_pred,
            'mask': mask,
        }

    def model_forward(self, batch):
        hi, mask = self.model(batch['X'], batch['lengths'])
        return {
            'hi': hi,
            'mask': mask,
        }

    def after_val_step(self, batch_idx, batch, result, meta):
        mask = result['output']['mask']
        meta['Y_true'].append(batch['Y'].masked_select(mask))
        meta['Y_pred'].append(result['output']['Y'].detach().masked_select(mask))
    
    def on_record_end(self, meta):
        meta = super().on_record_end(meta)
        meta['nt_summary'] = test4norm(meta['hi_dict'])
        return meta

@EVALUATOR_REGISTRY('MS')
class MSRTFEvaluator(BaseRTFEvaluator):
    '''MSRTF Model Evaluator'''

    def model_predict(self, batch):
        Y_pred, cls_mask = self.model.predict(batch['X'], batch['lengths'])
        return {
            'Y': Y_pred,
            'cls_mask': cls_mask,
        }
    
    def model_forward(self, batch):
        logits, hi, cls_mask, mfe_mask = self.model(batch['X'], batch['lengths'])
        return {
            'logits': logits,
            'hi': hi,
            'cls_mask': cls_mask,
            'mfe_mask': mfe_mask,
        }
    
    def after_val_step(self, batch_idx, batch, result, meta):
        cls_mask = result['output']['cls_mask']
        meta['Y_true'].append(batch['Y'].masked_select(cls_mask))
        meta['Y_pred'].append(result['output']['Y'].detach().masked_select(cls_mask))

    def after_record_step(self, batch_idx, batch, result, meta):
        hi_lengths = result['output']['mfe_mask'].sum(dim=-1)
        batch_hi = result['output']['hi'].detach().numpy()
        batch_UUT = batch['UUT']
        for UUT, hi, hi_length in zip(batch_UUT, batch_hi, hi_lengths):
            meta['hi_dict'][UUT] = hi[:hi_length]
