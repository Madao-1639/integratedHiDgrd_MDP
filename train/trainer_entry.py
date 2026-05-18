from .trainer import TRAINER_REGISTRY
from utils.preprocessing import read_preprocess_data
from utils.preprocessing import read_preprocess_data_NoiseAfterScale



def get_trainer(args, data=None, **trainer_kwargs):
    '''
    Initializes and returns a trainer or a list of trainers.

    Args:
        args: Configuration namespace containing data and trainer parameters.
        data (optional): Pre-loaded data passed to `read_preprocess_data`. Defaults to None.
        **trainer_kwargs: Additional keyword arguments to be passed to the trainer's constructor.

    Returns:
        For CV, returns a list of trainers for each fold.
        Otherwise, returns a single trainer instance.
    '''
    Trainer = TRAINER_REGISTRY[args.model_type]
    if args.k_fold > 0:
        return [
            Trainer(args, train_data=train_data, val_data=val_data, **trainer_kwargs)
            for train_data, val_data in read_preprocess_data(args, data=data)
        ]
    else:
        train_data, val_data = read_preprocess_data(args, data=data)
        return Trainer(args, train_data=train_data, val_data=val_data, **trainer_kwargs)

def get_trainer_NoiseAfterScale(args,data=None,**trainer_kwargs):
    Trainer = TRAINER_REGISTRY[args.model_type]
    if args.k_fold > 0:
        return [
            Trainer(args,train_data=train_data,val_data=val_data,**trainer_kwargs) 
            for train_data, val_data in read_preprocess_data_NoiseAfterScale(args,data=data)
        ]
    else:
        train_data, val_data = read_preprocess_data_NoiseAfterScale(args,data=data)
        return Trainer(args,train_data=train_data,val_data=val_data,**trainer_kwargs)