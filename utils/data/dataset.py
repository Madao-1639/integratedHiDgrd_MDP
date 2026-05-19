import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .collate_fn import custom_collate_fn, custom_collate_fn_ND, custom_RTF_collate_fn
from utils.registry import DATASET_REGISTRY



@DATASET_REGISTRY('Base')
class BaseDataset(Dataset):
    '''Base Dataset class for CMAPSS FD001 data.'''
    def __init__(self, data: pd.DataFrame, train: bool, args):
        '''Initialize the BaseDataset.

        Args:
            data (pd.DataFrame): Input data with columns 'UUT', 'time', and variable columns.
            train (bool): Whether this is a training dataset.
            args: Configuration containing dataset parameters.
        '''
        super().__init__()
        self.data = data
        self.train = train
        self.pos_label = args.pos_label
        self.collate_fn = custom_collate_fn

        self.parse_data()

        self.get_sample_indices()

    def parse_data(self):
        '''Extract UUT, time, features, and classification labels from the data.'''
        self.data = self.data.reset_index(drop=True) # Reset index to keep consistency
        self.UUT = self.data['UUT'].values
        self.time = self.data['time'].values

        grouped = self.data.groupby('UUT')
        self.grouped = grouped
        self.X = self.data.iloc[:,2:].values
        end_time = grouped.transform('max')['time']
        self.Y = (self.data['time']>end_time-self.pos_label).astype(int).values
        self.ls_dict = grouped['time'].max()

        # self.data = self.data.assign(label=(self.data['time']>end_time-self.pos_label).astype(int))

    def get_sample_indices(self):
        '''Set sample indices to all data indices (one sample per row).'''
        self.sample_indices = self.data.index

    def __getitem__(self, index):
        idx = self.sample_indices[index]
        UUT, t, x, y = self.UUT[idx], self.time[idx], self.X[idx], self.Y[idx]
        return UUT, t, x, y

    def __len__(self):
        return len(self.sample_indices)



@DATASET_REGISTRY('Base_ND')
class BaseDatasetND(BaseDataset):
    '''Base Dataset with N+1 consecutive samples for sequential input.'''
    def __init__(self, data, train, args):
        self.N = args.N
        super().__init__(data, train, args)
        self.collate_fn = custom_collate_fn_ND

    def get_sample_indices(self):
        '''Generate sample indices as N+1 consecutive time-step windows per UUT.'''
        sample_indices = []
        for UUT,group_indices in self.grouped.groups.items():
            end_time = self.ls_dict[UUT]
            sample_indices.extend(
                (t==self.N+1, t==end_time, # Start flag & End flag
                UUT,t,
                list(group_indices[t-i] for i in range(self.N+1,0,-1)),
                self.Y[group_indices[t-self.N-1:t]]    # Label for current time.
                ) for t in range(self.N+1,end_time+1)
            )
        self.sample_indices = sample_indices

    def __getitem__(self, index):        
        start, end, UUT, t, idx_tuple, Y = self.sample_indices[index]
        return start, end, UUT, t, [self.X[idx] for idx in idx_tuple], Y




# @DATASET_REGISTRY('TW')
# class TWDataset(BaseDataset):
#     def __init__(self, data, train, args):
#         ''' Input: data (Dataframe)
#         Create Time Windows(TW) of data.
#         A window, which is an element of one minibatch, has 3 elements:
#             UUT, t (time series the window covers) and x (features),
#             or 5 elements for train data, with y (labels) and start_sign (A bool to indicate the first window) added.
#         There is no overlap if 'sliding_offset' >= 'window_width'.'''
#         self.window_width = args.window_width
#         super().__init__(data, train, args)
#         self.collate_fn = custom_TW_collate_fn
    
#     def get_sample_indices(self):
#         '''Create Time Windows(TW) of data.
#         Return a list of (UUT,time,start_idx,end_idx) indicating a window.'''
#         sample_indices = []
#         for UUT, group_indices in self.grouped.groups.items():
#             end_time = self.ls_dict[UUT]
#             sample_indices.extend(
#                 (UUT,t,
#                     group_indices[t-self.window_width:t], # Window indices.
#                     self.Y[group_indices[t - 1]] # Label for current window.
#                 ) for t in range(self.window_width, end_time + 1)
#             )
#         self.sample_indices = sample_indices

#     def __getitem__(self, index):
#         UUT, t, indices, y =  self.sample_indices[index]
#         X = self.X[indices]  # Current window
#         return UUT, t, X, y



# @DATASET_REGISTRY('TW_ND')
# class TWDataset_ND(TWDataset):
#     def __init__(self, data, train, args):
#         self.N = args.N
#         super().__init__(data, train, args)
#         self.collate_fn = custom_collate_fn_ND

#     def get_sample_indices(self):
#         sample_indices = []
#         for UUT, group_indices in self.grouped.groups.items():
#             end_time = self.ls_dict[UUT]
#             sample_indices.extend(
#                 (
#                     t == self.window_width + self.N, t == end_time,
#                     UUT, t,
#                     list(group_indices[t-i-self.window_width:t-i]
#                     for i in range(self.N,-1,-1)),
#                     self.Y[group_indices[t-self.N-1:t]]
#                 ) for t in range(self.window_width + self.N, end_time + 1)
#             )
#         self.sample_indices = sample_indices

#     def __getitem__(self, index):        
#         start, end, UUT, t, indices_tuple, Y = self.sample_indices[index]
#         return start, end, UUT, t, [self.X[indices] for indices in indices_tuple], Y



@DATASET_REGISTRY('RTF')
class RTFDataset(BaseDataset):
    '''Run-to-Failure Dataset class for CMAPSS FD001 data.
    Each sample corresponds to one complete UUT run-to-failure trajectory.
    '''

    def __init__(self, data, train, args):
        super().__init__(data, train, args)
        self.collate_fn = custom_RTF_collate_fn
    def get_sample_indices(self):
        '''Set sample indices to unique UUT identifiers (one sample per UUT).'''
        self.sample_indices = list(self.grouped.groups.keys())

    def __getitem__(self, index):
        UUT = self.sample_indices[index]
        indice = self.grouped.groups[UUT]
        t, X, Y = self.time[indice], self.X[indice], self.Y[indice]
        return UUT, t, X, Y
