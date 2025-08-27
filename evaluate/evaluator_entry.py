from .evaluator import BaseRTFEvaluator,BaseTWEvaluator,SCEvaluator,IntegratedEvaluator,MSRTFEvaluator#,MSTWEvaluator
from utils.preprocessing import read_preprocess_data
from utils.preprocessing import read_preprocess_data_NoiseAfterScale

def select_evaluator_by_type(model_type, data_type=None):
    if model_type == 'Base':
        if data_type == 'RTF':
            return BaseRTFEvaluator
        elif data_type == 'TW':
            return BaseTWEvaluator
    elif model_type == 'SC':
        return SCEvaluator
    elif model_type == 'Integrated':
        return IntegratedEvaluator
    elif model_type == 'MS':
        if data_type == 'RTF':
            return MSRTFEvaluator
        # elif data_type == 'TW':
        #     return MSTWEvaluator

def get_evaluator(args,data=None,**evaluator_kwargs):
    '''
    Initializes and returns an evaluator.

    Args:
        args: Configuration namespace containing data and evaluator parameters.
        data (optional): Pre-loaded data passed to `read_preprocess_data`. Defaults to None.
        **evaluator_kwargs: Additional keyword arguments to be passed to the evaluator's constructor.

    Returns:
        an evaluator instance.
    '''
    Evaluator = select_evaluator_by_type(args.model_type,args.data_type)
    train_data, val_data = read_preprocess_data(args,data=data)
    return Evaluator(args,train_data=train_data,val_data=val_data,**evaluator_kwargs)

def get_evaluator_NoiseAfterScale(args,data=None,**evaluator_kwargs):
    Evaluator = select_evaluator_by_type(args.model_type,args.data_type)
    train_data, val_data = read_preprocess_data_NoiseAfterScale(args,data=data)
    return Evaluator(args,train_data=train_data,val_data=val_data,**evaluator_kwargs)