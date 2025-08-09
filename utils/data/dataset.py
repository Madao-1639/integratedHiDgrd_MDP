from torch.utils.data import Dataset,IterableDataset

class BaseDataset(Dataset):
    def __init__(self, data, train, args):
        ''' Base `Dataset` class for CMAPSS FD001 Dataset.
        Input a Dataframe with columns ('UUT','time',...variables).'''
        super().__init__()
        self.data = data
        self.train = train
        self.var_columns = self.data.columns[2:]

        self.task = args.task
        self.pos_label = args.pos_label
        self.get_label()

        grouped = self.data.groupby('UUT')
        self.grouped = grouped
        self.ls_dict = grouped['time'].max()

        self.get_sample_indice()

    def get_label(self):
        end_time = self.data.groupby('UUT')['time'].transform('max')
        if self.task == 'cls':
            self.data = self.data.assign(label=(self.data['time']>end_time-self.pos_label).astype(int))
        else:
            self.data = self.data.assign(label = end_time-self.data['time']+1)

    def get_sample_indice(self):
        self.sample_indice = self.data.index

    def __getitem__(self, index):
        idx = self.sample_indice[index]
        UUT, t, *data, y = self.data.loc[idx]
        return UUT, t, data, y

    def __len__(self):
        return len(self.sample_indice)



class BaseDataset_ND(BaseDataset):
    def __init__(self, data, train, args):
        self.N = args.N
        super().__init__(data,train,args)

    def get_sample_indice(self):
        sample_indice = []
        for UUT,group_indice in self.grouped.groups.items():
            end_time = self.ls_dict[UUT]
            sample_indice.extend(
                (t==self.N+1, t==end_time, # Start flag & End flag
                UUT,t,
                list(group_indice[t-i] for i in range(self.N+1,0,-1)),
                self.data.loc[group_indice[t-self.N-1:t],'label'].values    # Label for current time.
                ) for t in range(self.N+1,end_time+1)
            )
        self.sample_indice = sample_indice

    def __getitem__(self, index):        
        start, end, UUT, t, indice, y = self.sample_indice[index]
        return start, end, UUT, t, (self.data.loc[idx,self.var_columns].values for idx in indice), y



# class BaseDataset_ND_F(BaseDataset_ND):
#     def get_sample_indice(self):
#         sample_indice = []
#         for UUT,group_indice in self.grouped.groups.items():
#             end_time = self.ls_dict[UUT]
#             sample_indice.extend(
#                 (UUT,t,
#                 *(group_indice[t-i] for i in range(self.N+1,0,-1)),
#                 group_indice[-1],
#                 self.data.loc[group_indice[t-1],'label']
#                 ) for t in range(self.N+1,end_time+1)
#             )
#         self.sample_indice = sample_indice



class TWDataset(BaseDataset):
    def __init__(self, data, train, args):
        ''' Input: data (Dataframe)
        Create Time Windows(TW) of data.
        A window, which is an element of one minibatch, has 3 elements:
            UUT, t (time series the window covers) and x (features),
            or 5 elements for train data, with y (labels) and start_sign (A bool to indicate the first window) added.
        There is no overlap if 'sliding_offset' >= 'window_width'.'''
        self.window_width = args.window_width
        super().__init__(data, train,args)
    
    def get_sample_indice(self):
        '''Create Time Windows(TW) of data.
        Return a list of (UUT,time,start_idx,end_idx) indicating a window.'''
        sample_indice = []
        for UUT,group_indice in self.grouped.groups.items():
            end_time = self.ls_dict[UUT]
            sample_indice.extend(
                (UUT,t,
                    group_indice[t-self.window_width:t], # Window indice.
                    self.data.loc[group_indice[t-1],'label'] # Label for current window.
                ) for t in range(self.window_width,end_time+1)
            )
        self.sample_indice = sample_indice
    
    def __getitem__(self, index):
        UUT, t, indice, y =  self.sample_indice[index]
        data = self.data.loc[indice,self.var_columns].values  # Current window
        return UUT, t, data, y



class TWDataset_ND(TWDataset):
    def __init__(self, data, train, args):
        self.N = args.N
        super().__init__(data,train,args)

    def get_sample_indice(self):
        sample_indice = []
        for UUT,group_indice in self.grouped.groups.items():
            end_time = self.ls_dict[UUT]
            sample_indice.extend(
                (
                    t==self.window_width+self.N, t==end_time,
                    UUT,t,
                    list(group_indice[t-i-self.window_width:t-i]
                    for i in range(self.N,-1,-1)),
                    self.data.loc[group_indice[t-self.N-1:t],'label'].values
                ) for t in range(self.window_width+self.N,end_time+1)
            )
        self.sample_indice = sample_indice

    def __getitem__(self, index):        
        start, end, UUT, t, indice_tuple, y = self.sample_indice[index]
        return start, end, UUT, t, (self.data.loc[indice,self.var_columns].values for indice in indice_tuple), y



# class TWDataset_ND_F(TWDataset_ND):
#     def get_sample_indice(self):
#         sample_indice = []
#         for UUT,group_indice in self.grouped.groups.items():
#             end_time = self.ls_dict[UUT]
#             sample_indice.extend(
#                 (UUT,t,
#                     *(group_indice[t-i-self.window_width:t-i]
#                     for i in range(self.N,-1,-1)),
#                     group_indice[-self.window_width:], # Failure window indice.
#                     self.data.loc[group_indice[t-1],'label']
#                 ) for t in range(self.window_width+self.N,end_time+1)
#             )
#         self.sample_indice = sample_indice



class RTFDataset(IterableDataset):
    def __init__(self, data, train, args):
        super().__init__()
        self.data = data
        self.train = train
        self.var_columns = self.data.columns[2:]

        self.task = args.task
        self.pos_label = args.pos_label
        self.get_label()

        grouped = self.data.groupby('UUT')
        self.grouped = grouped
        self.ls_dict = grouped['time'].max()

    def get_label(self):
        end_time = self.data.groupby('UUT')['time'].transform('max')
        if self.task == 'cls':
            self.data = self.data.assign(label=(self.data['time']>end_time-self.pos_label).astype(int))
        else:
            self.data = self.data.assign(label = end_time-self.data['time']+1)

    def __iter__(self):
        for UUT, grouped_data in self.grouped:
            t = grouped_data['time'].values
            y = grouped_data['label'].values
            data = grouped_data[self.var_columns].values
            yield UUT, t, data, y



class RTFTWDataset(RTFDataset):
    def __init__(self, data, train, args):
        self.window_width = args.window_width 
        super().__init__(data, train, args)

    def __iter__(self):
        for UUT, grouped_data in self.grouped:
            grouped_aux = grouped_data.iloc[self.window_width-1:]
            t = grouped_aux['time'].values
            y = grouped_aux['label'].values
            data = []
            end_time = self.ls_dict[UUT]
            for window_time in range(self.window_width,end_time+1):
                window_data = grouped_data.loc[window_time-self.window_width:window_time,self.var_columns].values
                data.append(window_data)
            yield UUT, t, data, y