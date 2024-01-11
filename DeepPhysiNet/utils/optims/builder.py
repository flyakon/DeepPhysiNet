import torch.optim as optim
import torch
from bisect import bisect_right
from .lr_schedule_utils import WarmupMultiStepLR,WarmupStepLR
optim_dict={'Adam':optim.Adam,
            'SGD':optim.SGD}

def build_optim(name='Adam',**kwargs):
    if name in optim_dict.keys():
        return optim_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in availables values.'.format(name))





lr_schedule_dict={'stepLR':optim.lr_scheduler.StepLR,
                  'WarmupMultiStepLR':WarmupMultiStepLR,
                  'WarmupStepLR':WarmupStepLR,
                  'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR
                  }
def build_lr_schedule(name='stepLR',**kwargs):
    if name in lr_schedule_dict.keys():
        return lr_schedule_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in availables values.'.format(name))