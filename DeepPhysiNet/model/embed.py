'''
@Project : deep-downscale 
@File    : embed.py
@Author  : Wenyuan Li
@Date    : 2022/2/12 14:35 
@Desc    :  some codes are reffered from Informer: https://github.com/zhouhaoyi/Informer2020.git
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepPhysiNet.utils.position_encoding import SineCosPE
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x



class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.time_embending=SineCosPE(input_dim=1,include_input=False,N_freqs=int(d_model/2))

    def forward(self, x,forecast_h,learnable_token):
        x = self.value_embedding(x)
        x=torch.cat([learnable_token,x],dim=1)
        x=x+self.position_embedding(x)+self.time_embending(forecast_h)
        return x