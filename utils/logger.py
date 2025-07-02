import os
from collections import defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, args, log_name: "str" = None, comment: "str" = None, **writer_kwargs):
        if not log_name:
            log_name = args.model_name + '_' + args.time_str
        if comment:
            log_name = log_name + '_' + comment
        self.log_path = os.path.join(args.log_path, log_name)
        self.writer = SummaryWriter(self.log_path, **writer_kwargs)
        self.scalar_dict = defaultdict(list)
        self.scalars_dict = defaultdict(lambda: defaultdict(list))
        self.histogram_dict = defaultdict(list)
        self.normaltest_dict = defaultdict(list)
        # self.result_dir = args.result_dir
        self.checkpoint_path = args.checkpoint_path

    def record_scalar(self, tag, value):
        self.scalar_dict[tag].append(value)
    
    def record_scalars(self, main_tag, tag, value):
        self.scalars_dict[main_tag][tag].append(value)

    def record_histogram(self, tag, value):
        self.histogram_dict[tag].append(value)

    def record_normaltest(self,test_name,test_result):
        self.normaltest_dict[test_name].extend(test_result)

    def save_scalar(self, global_step,):
        for tag in list(self.scalar_dict.keys()):
            scaler = sum(self.scalar_dict[tag]) / len(self.scalar_dict[tag])
            del self.scalar_dict[tag]
            self.writer.add_scalar(tag, scaler, global_step)

    def save_scalars(self, global_step):
        for main_tag in list(self.scalars_dict.keys()):
            tag_scalar_dict = {}    
            for tag in self.scalars_dict[main_tag].keys():
                tag_scalar_dict[tag] = sum(self.scalars_dict[main_tag][tag]) / len(self.scalars_dict[main_tag][tag])
            del self.scalars_dict[main_tag]
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def save_histogram(self, global_step,):
        for tag,values in list(self.histogram_dict.items()):
            self.writer.add_histogram(tag, values, global_step)
            del self.histogram_dict[tag]

    def save_metrics(self,global_step,):
        self.save_scalar(global_step,)
        self.save_scalars(global_step,)
        self.save_histogram(global_step,)

    def save_checkpoint(self, model, epoch, step=0):
        checkpoint_name = f'{epoch:03d}_{step:05d}.pth'
        model_fp = os.path.join(self.checkpoint_path, checkpoint_name)
        torch.save(model.state_dict(), model_fp)

    def __exit__(self):
        self.writer.close()
