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

def get_trainer(args,**trainer_kwargs):
    Trainer = select_trainer_by_type(args.model_type,args.data_type)
    for train_data,val_data,test_data in read_preprocess_data(args):
        yield Trainer(args,train_data=train_data,val_data=val_data,**trainer_kwargs)

def get_trainer_NoiseAfterScale(args,**trainer_kwargs):
    Trainer = select_trainer_by_type(args.model_type,args.data_type)
    for train_data,val_data,test_data in read_preprocess_data_NoiseAfterScale(args):
        yield Trainer(args,train_data=train_data,val_data=val_data,**trainer_kwargs)