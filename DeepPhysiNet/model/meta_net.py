'''
@Project : deep-downscale 
@File    : meta_net.py
@Author  : Wenyuan Li
@Date    : 2022/7/3 18:14 
@Desc    :  
'''

import torch.nn as nn
import torch
from .transformer_net import TransformerNet

class MetaNet(nn.Module):
    def __init__(self,meta_cfg):
        super(MetaNet, self).__init__()
        self.meta_cfg=meta_cfg
        self.model=TransformerNet(**meta_cfg)

    def forward(self,x,forecast_h):
        return self.model.forward(x,forecast_h)

