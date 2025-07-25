import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LogTransformer:
    def __init__(self,_delta=1):
        self.delta = _delta
    def fit(self,data):
        '''
        Calculate the phi value based on the minimum value of input data.
        '''
        self.phi = - data.min() + self.delta
    def transform(self,data):
        '''
        Apply the logarithmic transformation to data.
        '''
        return np.log(data+self.phi)
    def fit_transform(self,data):
        '''
        Fit the transformer to the data and then transform it.
        '''
        self.fit(data)
        return self.transform(data)

def read_data(data_fp, drop_vars):
    '''
    1. Read data as a Dataframe.
    2. Drop condition columns and duplicate variables.

    Args:
        data_fp (str): File path of the data.
        drop_vars (list or None): List of variables to drop.

    Returns:
        pd.DataFrame: Processed data.
    '''
    var_cols = list(range(5,26))
    usecols = [0,1] + var_cols # Drop condition columns
    dtype_dict = {_:'float32' for _ in var_cols} # Specify dtypes
    dtype_dict[0] = dtype_dict[1] = 'int'
    data = pd.read_csv(data_fp, usecols=usecols, dtype=dtype_dict, header=None, sep=r'\s+',)
    columns = ['UUT','time'] + list(range(1,22)) # 21 measurements
    data.columns = columns
    if drop_vars is not None: # Filter variables
        data.drop(drop_vars,axis=1,inplace=True)
    data.sort_values(['UUT','time'],inplace=True) # Keep order
    return data

def add_noise(data, noise_type = 'gaussian', noise_param = 0.1):
    '''
    Add noise to raw data.

    Args:
        data (pd.DataFrame): Raw data with columns of 'UUT', 'time', var1, var2, ... ('label')
        noise_type (str): Type of noise to add. Options are 'gaussian' or 'white gaussian'.
        noise_param (float): Parameter for the noise.

    Returns:
        pd.DataFrame: Synthetic data with added noise.
    '''
    var_cols = [col for col in data.columns if col not in ('UUT','time','label')]
    if noise_type == 'gaussian': # Add Gaussian Noise
        noise = (noise_param*np.random.randn(len(data),len(var_cols))).astype('float32')
        newdata = data
        newdata.loc[:,var_cols] += noise
    elif noise_type == 'white gaussian': # Add White Gaussian Noise:
        grouped = data.groupby('UUT')
        newdata = []
        for UUT,sample in grouped:
            px_sqrt = np.sqrt(np.power(sample.loc[:,var_cols].values,2).mean(axis=0))
            pn_sqrt = (px_sqrt*(10**(-noise_param/20.))).reshape(1,-1)
            noise = (pn_sqrt*np.random.randn(len(sample),len(var_cols))).astype('float32')
            sample.loc[:,var_cols] += noise
            newdata.append(sample)
        newdata = pd.concat(newdata)
    return newdata

def gen_cv_data(data, k_fold = 5):
    '''Generate Cross-Validation(cv) Data.'''
    grouped = data.groupby('UUT')
    fold_size = math.ceil(len(grouped)/k_fold)
    all_UUTs = list(grouped.groups)
    np.random.shuffle(all_UUTs)
    for k in range(k_fold):
        val_UUTs = all_UUTs[k*fold_size:(k+1)*fold_size]
        filter_bool = data['UUT'].isin(val_UUTs)
        train_data = data[~filter_bool]
        val_data = data[filter_bool]
        yield train_data, val_data

def gen_loo_data(data, val_ratio = 0.2):
    '''Generate Leave-One-Out(loo) Data.'''
    grouped = data.groupby('UUT')
    val_UUTs = np.random.choice(list(grouped.groups),
            size=int(val_ratio*len(grouped)),
        ) # Stratified random sampling: p=grouped.size()/len(data)
    filter_bool = data['UUT'].isin(val_UUTs)
    train_data = data[~filter_bool]
    val_data = data[filter_bool]
    return train_data, val_data

def select_scaler_by_type(scaler_type = 'Standard'):
    '''Select a scaler based on the given type.'''
    if scaler_type == 'Standard':
        return StandardScaler()
    elif scaler_type == 'MinMax':
        return MinMaxScaler()

def apply_transformations(data_split, scaler_type = 'Standard', log_transform = False):
    '''
    Apply scaling and logarithmic transformation to the data split.

    Args:
        data_split (tuple): (train_data, val_data, test_data)
        scaler_type (str): Type of scaler.
        log_transform (bool): Whether to apply logarithmic transformation.

    Returns:
        tuple: Transformed (train_data, val_data, test_data)
    '''
    train_data, val_data, test_data = data_split
    var_cols = [col for col in train_data.columns if col not in ('UUT','time','label')]
    if scaler_type is not None:
        scaler = select_scaler_by_type(scaler_type)
        train_data.loc[:,var_cols] = scaler.fit_transform(train_data.loc[:,var_cols])
        if val_data is not None:
            val_data.loc[:,var_cols] = scaler.transform(val_data.loc[:,var_cols])
        if test_data is not None:
            test_data.loc[:,var_cols] = scaler.transform(test_data.loc[:,var_cols])
    if log_transform:
        log_transformer = LogTransformer()
        train_data.loc[:,var_cols] = log_transformer.fit_transform(train_data.loc[:,var_cols])
        if val_data is not None:
            val_data.loc[:,var_cols] = log_transformer.transform(val_data.loc[:,var_cols])
        if test_data is not None:
            test_data.loc[:,var_cols] = log_transformer.transform(test_data.loc[:,var_cols])
    return train_data, val_data, test_data


def read_preprocess_data(args, data=None):
    '''
    Read data from file, apply preprocessing, and generate data splits (train_data, val_data, test_data).

    Args:
        args: Configuration namespace containing preprocessing and splitting parameters
        data (Optional): Pre-loaded data. If None, data will be loaded from file.
    
    Yields:
        Processed data splits based on specified configuration.
    '''
    if data is None:
        data = read_data(args.train_fp, args.drop_vars)
    if args.add_noise:
        data = add_noise(data,args.noise_type,args.noise_param)

    if 0 < args.test_ratio < 1:
        data, test_data = gen_loo_data(data,args.test_ratio)
    else:
        test_data = None
    if args.k_fold > 0:
        for train_data, val_data in gen_cv_data(data,args.k_fold):
            yield apply_transformations((train_data, val_data, None), args.scaler_type, args.log_transform)
    else:
        if 0 < args.val_ratio < 1:
            train_data, val_data = gen_loo_data(data,args.val_ratio/(1-args.test_ratio))
        else:
            train_data = data
            val_data = None
        yield apply_transformations((train_data, val_data, test_data), args.scaler_type, args.log_transform)


def read_preprocess_data_NoiseAfterScale(args,data=None):
    if data is None:
        data = read_data(args.train_fp, args.drop_vars)
    if 0 < args.test_ratio < 1:
        data, test_data = gen_loo_data(data,args.test_ratio)
    else:
        test_data = None
    if args.k_fold > 0:
        for train_data, val_data in gen_cv_data(data,args.k_fold):
            if args.scaler_type is not None:
                scaler = select_scaler_by_type(args.scaler_type)
                train_data.iloc[:,2:] = scaler.fit_transform(train_data.iloc[:,2:])
                val_data.iloc[:,2:] = scaler.transform(val_data.iloc[:,2:])
            if args.add_noise:
                train_data = add_noise(train_data,args.noise_type,args.noise_param)
                val_data =  add_noise(val_data,args.noise_type,args.noise_param)
            if args.log_transform:
                log_transformer = LogTransformer()
                train_data.iloc[:,2:] = log_transformer.fit_transform(train_data.iloc[:,2:])
                val_data.iloc[:,2:] = log_transformer.transform(val_data.iloc[:,2:])
            yield train_data, val_data, None
    else:
        if 0 < args.val_ratio < 1:
            train_data, val_data = gen_loo_data(data,args.val_ratio/(1-args.test_ratio))
        else:
            train_data = data
            val_data = None
        if args.scaler_type is not None:
            scaler = select_scaler_by_type(args.scaler_type)
            train_data.iloc[:,2:] = scaler.fit_transform(train_data.iloc[:,2:])
            if val_data is not None:
                val_data.iloc[:,2:] = scaler.transform(val_data.iloc[:,2:])
            if test_data is not None:
                test_data.iloc[:,2:] = scaler.transform(test_data.iloc[:,2:])
        if args.add_noise:
            train_data = add_noise(train_data,args.noise_type,args.noise_param)
            if val_data is not None:
                val_data =  add_noise(val_data,args.noise_type,args.noise_param)
            if test_data is not None:
                test_data = add_noise(test_data,args.noise_type,args.noise_param)   
        if args.log_transform:
            log_transformer = LogTransformer()
            train_data.iloc[:,2:] = log_transformer.fit_transform(train_data.iloc[:,2:])
            if val_data is not None:
                val_data.iloc[:,2:] = log_transformer.transform(val_data.iloc[:,2:])
            if test_data is not None:
                test_data.iloc[:,2:] = log_transformer.transform(test_data.iloc[:,2:])
        yield train_data, val_data, test_data