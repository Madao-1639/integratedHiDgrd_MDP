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

        self.get_sample_indices()

    def get_label(self):
        end_time = self.data.groupby('UUT')['time'].transform('max')
        if self.task == 'cls':
            self.data = self.data.assign(label=(self.data['time']>end_time-self.pos_label).astype(int))
        else:
            self.data = self.data.assign(label = end_time-self.data['time']+1)

    def get_sample_indices(self):
        self.sample_indices = self.data.index

    def __getitem__(self, index):
        idx = self.sample_indices[index]
        UUT, t, *data, y = self.data.loc[idx]
        return UUT, t, data, y

    def __len__(self):
        return len(self.sample_indices)



class BaseDataset_ND(BaseDataset):
    def __init__(self, data, train, args):
        self.N = args.N
        super().__init__(data,train,args)

    def get_sample_indices(self):
        sample_indices = []
        for UUT,group_indices in self.grouped.groups.items():
            end_time = self.ls_dict[UUT]
            sample_indices.extend(
                (t==self.N+1, t==end_time, # Start flag & End flag
                UUT,t,
                list(group_indices[t-i] for i in range(self.N+1,0,-1)),
                self.data.loc[group_indices[t-self.N-1:t],'label'].values    # Label for current time.
                ) for t in range(self.N+1,end_time+1)
            )
        self.sample_indices = sample_indices

    def __getitem__(self, index):        
        start, end, UUT, t, indices, y = self.sample_indices[index]
        return start, end, UUT, t, (self.data.loc[idx,self.var_columns].values for idx in indices), y



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
    
    def get_sample_indices(self):
        '''Create Time Windows(TW) of data.
        Return a list of (UUT,time,start_idx,end_idx) indicating a window.'''
        sample_indices = []
        for UUT,group_indices in self.grouped.groups.items():
            end_time = self.ls_dict[UUT]
            sample_indices.extend(
                (UUT,t,
                    group_indices[t-self.window_width:t], # Window indices.
                    self.data.loc[group_indices[t-1],'label'] # Label for current window.
                ) for t in range(self.window_width,end_time+1)
            )
        self.sample_indices = sample_indices
    
    def __getitem__(self, index):
        UUT, t, indices, y =  self.sample_indices[index]
        data = self.data.loc[indices,self.var_columns].values  # Current window
        return UUT, t, data, y



class TWDataset_ND(TWDataset):
    def __init__(self, data, train, args):
        self.N = args.N
        super().__init__(data,train,args)

    def get_sample_indices(self):
        sample_indices = []
        for UUT,group_indices in self.grouped.groups.items():
            end_time = self.ls_dict[UUT]
            sample_indices.extend(
                (
                    t==self.window_width+self.N, t==end_time,
                    UUT,t,
                    list(group_indices[t-i-self.window_width:t-i]
                    for i in range(self.N,-1,-1)),
                    self.data.loc[group_indices[t-self.N-1:t],'label'].values
                ) for t in range(self.window_width+self.N,end_time+1)
            )
        self.sample_indices = sample_indices

    def __getitem__(self, index):        
        start, end, UUT, t, indices_tuple, y = self.sample_indices[index]
        return start, end, UUT, t, (self.data.loc[indices,self.var_columns].values for indices in indices_tuple), y



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