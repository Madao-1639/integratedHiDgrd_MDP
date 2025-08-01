from .trainer import BaseRTFTrainer,BaseTWTrainer,SCTrainer,IntegratedTrainer,MSRTFTrainer#,MSTWTrainer
from utils.preprocessing import read_preprocess_data
from utils.preprocessing import read_preprocess_data_NoiseAfterScale

def select_trainer_by_type(model_type, data_type=None):
    if model_type == 'Base':
        if data_type == 'RTF':
            return BaseRTFTrainer
        elif data_type == 'TW':
            return BaseTWTrainer
    elif model_type == 'SC':
        return SCTrainer
    elif model_type == 'Integrated':
        return IntegratedTrainer
    elif model_type == 'MS':
        if data_type == 'RTF':
            return MSRTFTrainer
        # elif data_type == 'TW':
        #     return MSTWTrainer

def get_trainer(args,data=None,**trainer_kwargs):
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
    Trainer = select_trainer_by_type(args.model_type,args.data_type)
    if args.k_fold > 0:
        return [
            Trainer(args,train_data=train_data,val_data=val_data,**trainer_kwargs)
            for train_data,val_data in read_preprocess_data(args,data=data)
        ]
    else:
        train_data, val_data = read_preprocess_data(args,data=data)
        return Trainer(args,train_data=train_data,val_data=val_data,**trainer_kwargs)

def get_trainer_NoiseAfterScale(args,data=None,**trainer_kwargs):
    Trainer = select_trainer_by_type(args.model_type,args.data_type)
    if args.k_fold > 0:
        return [
            Trainer(args,train_data=train_data,val_data=val_data,**trainer_kwargs) 
            for train_data, val_data in read_preprocess_data_NoiseAfterScale(args,data=data)
        ]
    else:
        train_data, val_data = read_preprocess_data_NoiseAfterScale(args,data=data)
        return Trainer(args,train_data=train_data,val_data=val_data,**trainer_kwargs)