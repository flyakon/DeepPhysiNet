'''
@Project : deep-downscale 
@File    : weights_loss.py
@Author  : Wenyuan Li
@Date    : 2022/7/11 17:29 
@Desc    :  
'''

import torch.nn as nn
import torch

class WeightSmoothL1Loss(nn.Module):
    def __init__(self,beta=0.1):
        super(WeightSmoothL1Loss, self).__init__()
        self.criterion=nn.SmoothL1Loss(beta=beta,reduction='none')

    def forward(self,input,target):
        loss=self.criterion.forward(input,target)
        loss=torch.mean(loss)
        return loss