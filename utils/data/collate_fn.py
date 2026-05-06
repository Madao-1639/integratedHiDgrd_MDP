import torch
from torch.nn.utils.rnn import pad_sequence



def _pack_data(batch_data):
    return torch.stack([torch.FloatTensor(data) for data in batch_data])

def custom_collate_fn(batch):
    batch_UUT, batch_t, batch_X, batch_Y = zip(*batch)
    batch_X = torch.FloatTensor(batch_X)
    batch_Y = torch.FloatTensor(batch_Y)
    return {
        'UUT': batch_UUT,
        't': batch_t,
        'X': batch_X,
        'Y': batch_Y,
    }

def custom_collate_fn_ND(batch):
    batch_start, batch_end, batch_UUT, batch_t, batch_multi_X, batch_multi_Y = zip(*batch)
    batch_start = torch.BoolTensor(batch_start)
    batch_end = torch.BoolTensor(batch_end)
    return {
        'start': batch_start,
        'end': batch_end,
        'UUT': batch_UUT,
        't': batch_t,
        'X': [torch.FloatTensor(batch_X) for batch_X in zip(*batch_multi_X)],
        'Y': [torch.FloatTensor(batch_y) for batch_y in zip(*batch_multi_Y)],
    }

def custom_TW_collate_fn(batch):
    batch_UUT, batch_t, batch_X, batch_Y = zip(*batch)
    batch_Y = torch.FloatTensor(batch_Y)
    return {
        'UUT': batch_UUT,
        't': batch_t,
        'X': _pack_data(batch_X),
        'Y': batch_Y,
    }

def custom_RTF_collate_fn(batch):
    '''
    Custom collate function for RTF data to convert batched data into PyTorch tensor format.

    This function takes a list of samples in a batch, where each sample is a tuple (UUT, t, X, Y).
    It stacks UUTs into a single batch tensor, while converting t, X, and Y into lists of tensors
    to handle variable-length or non-uniform shaped data.

    Args:
        batch (list): A list where each element is a tuple (UUT, t, X, Y).
            - UUT: Typically a scalar or fixed-shape numeric value.
            - t: Time series or other variable-length data.
            - X: Input feature data.
            - Y: Label or target data.

    Returns:
        tuple: A tuple containing four elements:
            - batch_UUT (torch.Tensor): Stacked tensor of UUTs.
            - batch_t (list[torch.Tensor]): List of converted tensors for t.
            - batch_X (list[torch.FloatTensor]): List of converted float tensors for X.
            - batch_Y (list[torch.FloatTensor]): List of converted float tensors for Y.
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