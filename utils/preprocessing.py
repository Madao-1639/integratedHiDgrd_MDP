import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.registry import SCALER_REGISTRY

SCALER_REGISTRY.register("Standard", StandardScaler)
SCALER_REGISTRY.register("MinMax", MinMaxScaler)

class LogTransformer:
    '''Logarithmic transformer that shifts data to be positive before applying log.'''

    def __init__(self, _delta: float = 1) -> None:
        '''Initialize the LogTransformer.

        Args:
            _delta (float): Small constant added to the shift value. Defaults to 1.
        '''
        self.delta = _delta
    def fit(self, data):
        '''
        Calculate the phi value based on the minimum value of input data.
        '''
        self.phi = - data.min() + self.delta
    def transform(self, data):
        '''
        Apply the logarithmic transformation to data.
        '''
        return np.log(data + self.phi)
    def fit_transform(self, data):
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
    var_cols = [var for var in range(1, 22) if not drop_vars or (drop_vars and var not in drop_vars)] # Filter variables
    usecols = [0,1] + [var + 4 for var in var_cols] # Drop condition columns
    dtype_dict = {_:'float64' for _ in var_cols} # Specify dtypes
    dtype_dict[0] = dtype_dict[1] = 'int'
    data = pd.read_csv(data_fp, usecols = usecols, dtype = dtype_dict, header = None, sep = r'\s+',)
    data.columns = ['UUT','time'] + var_cols
    data.sort_values(['UUT','time'], inplace = True) # Keep order
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
    var_cols = [col for col in data.columns if col not in ('UUT', 'time', 'label')]
    if noise_type == 'gaussian': # Add Gaussian Noise
        noise = (noise_param * np.random.randn(len(data), len(var_cols))).astype('float32')
        newdata = data
        newdata.loc[:, var_cols] += noise
    elif noise_type == 'white gaussian': # Add White Gaussian Noise:
        grouped = data.groupby('UUT')
        newdata = []
        for UUT, sample in grouped:
            px_sqrt = np.sqrt(np.power(sample.loc[:, var_cols].values, 2).mean(axis=0))
            pn_sqrt = (px_sqrt * (10 ** (-noise_param / 20.))).reshape(1, -1)
            noise = (pn_sqrt * np.random.randn(len(sample), len(var_cols))).astype('float32')
            sample.loc[:, var_cols] += noise
            newdata.append(sample)
        newdata = pd.concat(newdata)
    return newdata

def gen_cv_data(data, k_fold = 5):
    '''Generate Cross-Validation (CV) data splits.

    Args:
        data (pd.DataFrame): Input data with 'UUT' column for grouping.
        k_fold (int): Number of folds. Defaults to 5.

    Yields:
        tuple[pd.DataFrame, pd.DataFrame]: Train and validation data for each fold.
    '''
    grouped = data.groupby('UUT')
    fold_size = math.ceil(len(grouped) / k_fold)
    all_UUTs = list(grouped.groups)
    np.random.shuffle(all_UUTs)
    for k in range(k_fold):
        val_UUTs = all_UUTs[k * fold_size:(k + 1) * fold_size]
        filter_bool = data['UUT'].isin(val_UUTs)
        train_data = data[~filter_bool]
        val_data = data[filter_bool]
        yield train_data, val_data

def gen_loo_data(data, val_ratio = 0.2):
    '''Generate Leave-One-Out (LOO) data split.

    Args:
        data (pd.DataFrame): Input data with 'UUT' column for grouping.
        val_ratio (float): Fraction of units for validation. Defaults to 0.2.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and validation data.
    '''
    grouped = data.groupby('UUT')
    val_UUTs = np.random.choice(list(grouped.groups),
            size=int(val_ratio * len(grouped))
        ) # Stratified random sampling: p=grouped.size()/len(data)
    filter_bool = data['UUT'].isin(val_UUTs)
    train_data = data[~filter_bool]
    val_data = data[filter_bool]
    return train_data, val_data

def apply_transformations(args, train_data, val_data=None):
    '''
    Apply scaling and logarithmic transformation to train_data and val_data.

    Args:
        args: Namespace or object containing transformation parameters
        train_data (pd.DataFrame): The training dataset containing features and possibly labels.
        val_data (pd.DataFrame, optional): The validation dataset to be transformed using the same scaler/transformer as train_data. Defaults to None.

    Returns:
        tuple: Transformed train_data and val_data.
    '''
    if args.add_noise:
        train_data = add_noise(train_data, args.noise_type, args.noise_param)
        if val_data is not None:
            val_data = add_noise(val_data, args.noise_type, args.noise_param)
    var_cols = [col for col in train_data.columns if col not in ('UUT', 'time', 'label')]
    if args.scaler_type is not None:
        scaler = SCALER_REGISTRY[args.scaler_type]()
        train_data.loc[:, var_cols] = scaler.fit_transform(train_data.loc[:, var_cols])
        if val_data is not None:
            val_data.loc[:, var_cols] = scaler.transform(val_data.loc[:, var_cols])
    if args.log_transform:
        log_transformer = LogTransformer()
        train_data.loc[:, var_cols] = log_transformer.fit_transform(train_data.loc[:, var_cols])
        if val_data is not None:
            val_data.loc[:, var_cols] = log_transformer.transform(val_data.loc[:, var_cols])
    return train_data, val_data

def read_preprocess_data(args, data=None):
    '''
    Read data from file, split data and apply preprocessing to data.
    Note that there is no need to apply transformations to test_data, and data splitting should be done before transformations to (i) exclude test_data from parameter tuning and (ii) keep consistency of test_data for parameter tuning and testing, due to the randomness involved in noise addition.

    Args:
        args: Configuration namespace containing preprocessing and splitting parameters.
        data (Optional): Pre-loaded data. If None, data will be loaded from file. Defaults to None.
    
    Returns:
        For CV, returns a list of tuples, each containing (train_data, val_data) for each fold, after transformations.
        Otherwise, returns processed train_data and val_data based on specified configuration.
    '''
    if data is None:
        data = read_data(args.data_fp, args.drop_vars)
    if 0 < args.test_ratio < 1:
        data, test_data = gen_loo_data(data, args.test_ratio)

    if args.k_fold > 0:
        return [
            apply_transformations(args, train_data, val_data)
            for train_data, val_data in gen_cv_data(data, args.k_fold)
        ]
    else:
        if 0 < args.val_ratio < 1:
            train_data, val_data = gen_loo_data(data, args.val_ratio / (1 - args.test_ratio))
        else:
            train_data = data
            val_data = None
        return apply_transformations(args, train_data, val_data)

def apply_transformations_NoiseAfterScale(args, train_data, val_data = None):
    var_cols = [col for col in train_data.columns if col not in ('UUT','time','label')]
    if args.scaler_type is not None:
        scaler = SCALER_REGISTRY[args.scaler_type]()
        train_data.loc[:,var_cols] = scaler.fit_transform(train_data.loc[:,var_cols])
        if val_data is not None:
            val_data.loc[:,var_cols] = scaler.transform(val_data.loc[:,var_cols])
    if args.add_noise:
        train_data = add_noise(train_data,args.noise_type,args.noise_param)
        if val_data is not None:
            val_data =  add_noise(val_data,args.noise_type,args.noise_param)
    if args.log_transform:
        log_transformer = LogTransformer()
        train_data.loc[:,var_cols] = log_transformer.fit_transform(train_data.loc[:,var_cols])
        if val_data is not None:
            val_data.loc[:,var_cols] = log_transformer.transform(val_data.loc[:,var_cols])
    return train_data, val_data

def read_preprocess_data_NoiseAfterScale(args,data=None):
    if data is None:
        data = read_data(args.data_fp, args.drop_vars)
    if 0 < args.test_ratio < 1:
        data, test_data = gen_loo_data(data,args.test_ratio)
    if args.k_fold > 0:
        return [
            apply_transformations_NoiseAfterScale(args, train_data, val_data) 
            for train_data, val_data in gen_cv_data(data,args.k_fold)
        ]
    else:
        if 0 < args.val_ratio < 1:
            train_data, val_data = gen_loo_data(data,args.val_ratio/(1-args.test_ratio))
        else:
            train_data = data
            val_data = None
        return apply_transformations_NoiseAfterScale(args, train_data, val_data)