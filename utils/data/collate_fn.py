import torch
from torch.nn.utils.rnn import pad_sequence



def _pack_data(batch_data):
    '''Pack a batch of data samples into a single tensor by stacking them.'''
    return torch.stack([torch.FloatTensor(data) for data in batch_data])

def custom_collate_fn(batch):
    '''
    Custom collate function to process a batch of data samples for model training.
    
    This function unpacks a batch of samples, converts them into appropriate tensor formats,
    and returns a dictionary containing the processed batch data ready for model input.

    Args:
        batch: A list of tuples, where each tuple contains 4 elements:
            - UUT: Unit Under Test identifier
            - t: Time series data
            - X: Input feature data (will be stacked into a tensor)
            - Y: Target/label data (will be converted to FloatTensor)
    
    Returns:
        dict: A dictionary containing the processed batch with the following keys:
            - UUT (tuple): Batch of UUTs
            - t (tuple): Batch of time series
            - X (torch.Tensor): Stacked tensor of input feature data with shape (batch_size, *)
            - Y (torch.FloatTensor): Float tensor of target/label data with shape (batch_size,)
    '''
    batch_UUT, batch_t, batch_X, batch_Y = zip(*batch)
    batch_X = _pack_data(batch_X)
    batch_Y = torch.FloatTensor(batch_Y)
    return {
        'UUT': batch_UUT,
        't': batch_t,
        'X': batch_X,
        'Y': batch_Y,
    }

def custom_collate_fn_ND(batch):
    '''
    Custom collate function to process a batch of multi-dimensional data samples for model training.
    
    This function unpacks a batch of samples containing start/end flags, unit identifiers, time steps,
    multiple feature sequences, and multiple target sequences. It converts them into appropriate tensor
    formats and returns a dictionary containing the processed batch data ready for model input.
    
    Args:
        batch: A list of tuples, where each tuple contains 6 elements:
            - start (bool): Flag indicating if this is the start of a sequence
            - end (bool): Flag indicating if this is the end of a sequence
            - UUT: Unit Under Test identifier
            - t: Time series
            - multi_X: A list/tuple of feature data with shape (batch_size, *)
            - multi_Y: A list/tuple of target/label data with shape (batch_size,)
    
    Returns:
        dict: A dictionary containing the processed batch with the following keys:
            - start (torch.BoolTensor): Batch of sequence start flags
            - end (torch.BoolTensor): Batch of sequence end flags
            - UUT (tuple): Batch of UUTs
            - t (tuple): Batch of time series
            - X (list): List of stacked tensors with shape (batch_size, *)
            - Y (list): List of float tensors with shape (batch_size,)
    '''
    batch_start, batch_end, batch_UUT, batch_t, batch_multi_X, batch_multi_Y = zip(*batch)
    batch_start = torch.BoolTensor(batch_start)
    batch_end = torch.BoolTensor(batch_end)
    return {
        'start': batch_start,
        'end': batch_end,
        'UUT': batch_UUT,
        't': batch_t,
        'X': [_pack_data(batch_X) for batch_X in zip(*batch_multi_X)],
        'Y': [torch.FloatTensor(batch_y) for batch_y in zip(*batch_multi_Y)],
    }

# def custom_TW_collate_fn(batch):
#     batch_UUT, batch_t, batch_X, batch_Y = zip(*batch)
#     batch_Y = torch.FloatTensor(batch_Y)
#     return {
#         'UUT': batch_UUT,
#         't': batch_t,
#         'X': _pack_data(batch_X),
#         'Y': batch_Y,
#     }

def custom_RTF_collate_fn(batch):
    '''
    Custom collate function for RTF data to convert batched data into PyTorch tensor format.

    This function takes a list of samples in a batch, where each sample is a tuple (UUT, t, X, Y).
    It converts t, X, and Y into packed tensors with equal lengths to handle variable-length or non-uniform shaped data.

    Args:
        batch (list): A list where each element is a tuple (UUT, t, X, Y)
            - UUT: UUT numbers of samples
            - t: Time series
            - X: Input feature data
            - Y: Label or target data

    Returns:
        dict: A dictionary containing the processed batch with the following keys:
            - batch_UUT (torch.Tensor): Stacked tensor of UUTs
            - batch_t (list): Converted tensor for t with shape (batch_size, max_len,)
            - batch_X (list): Converted float tensor for X with shape (batch_size, max_len, *)
            - batch_Y (list): Converted float tensor for Y with shape (batch_size, max_len,)
    ''' 
    batch_UUT, batch_t, batch_X, batch_Y = zip(*batch)
    lengths = torch.as_tensor([t.shape[0] for t in batch_t])
    batch_t = pad_sequence([torch.FloatTensor(t) for t in batch_t], batch_first = True)
    batch_X = pad_sequence([torch.FloatTensor(X) for X in batch_X], batch_first = True)
    batch_Y = pad_sequence([torch.FloatTensor(Y) for Y in batch_Y], batch_first = True)
    return {
        'UUT': batch_UUT,
        't': batch_t,
        'X': batch_X,
        'Y': batch_Y,
        'lengths': lengths,
    }