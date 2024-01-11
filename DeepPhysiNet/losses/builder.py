

import torch
import torchvision
import torch.nn as nn
from .weights_loss import WeightSmoothL1Loss

losses_dict={'CrossEntropyLoss':nn.CrossEntropyLoss,
             'L1Loss':nn.L1Loss,
             'MSELoss':nn.MSELoss,
             'WeightSmoothL1Loss':WeightSmoothL1Loss
             }


def builder_loss(name='CrossEntropyLoss',**kwargs):

    if name in losses_dict.keys():
        return losses_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in availables values.'.format(name))