'''
@Project : deep-downscale 
@File    : feature_net.py
@Author  : Wenyuan Li
@Date    : 2022/2/11 17:56 
@Desc    :  Network for extracting features of input variables
'''
import torch.nn as nn
import torch
import os
import numpy as np
from .backbone.builder import  build_backbone
from .meta_net import MetaNet
from .variable_net import VariableNet
import torch.nn.functional as F

class PhysicsNet(torch.nn.Module):
    def __init__(self,meta_cfg:dict,net_cfg:dict):
        super(PhysicsNet, self).__init__()
        in_channels=net_cfg['in_channels']
        hidden_channels=net_cfg['hidden_channels']
        token_num = net_cfg['learnable_token_num']

        self.meta_net=MetaNet(meta_cfg)
        self.U_net = VariableNet(token_num,in_channels,hidden_channels)
        self.V_net = VariableNet(token_num,in_channels,hidden_channels)
        self.P_net = VariableNet(token_num,in_channels,hidden_channels)
        self.T_net = VariableNet(token_num,in_channels,hidden_channels)
        self.rio_net = VariableNet(token_num,in_channels,hidden_channels)
        self.q_net=VariableNet(token_num,in_channels,hidden_channels)

        self.tanh=nn.Tanh()
        self.net_dict={'u':self.U_net,
                       'v':self.V_net,
                       'p':self.P_net,
                       'T':self.T_net,
                       'q':self.q_net,
                       'rio': self.rio_net,
                       }

    def forward(self,field_x, coord_x,coord_data,forecast_h):
        field_features=self.meta_net.forward(field_x,forecast_h)
        # U=self.U_net(field_features,coord_x,coord_data[:,0:1],forecast_h)
        # V=self.V_net(field_features,coord_x,coord_data[:,1:2],forecast_h)
        # P=self.P_net(field_features,coord_x,coord_data[:,2:3],forecast_h)
        # T=self.T_net(field_features,coord_x,coord_data[:,3:4],forecast_h)
        # q=self.q_net(field_features,coord_x,coord_data[:,4:5],forecast_h)
        # rio = self.rio_net(field_features,coord_x,coord_data[:,5:6],forecast_h)
        U = self.U_net(field_features, coord_x, coord_data,coord_data[:,0:1], forecast_h)
        V = self.V_net(field_features, coord_x, coord_data,coord_data[:,1:2], forecast_h)
        P = self.P_net(field_features, coord_x, coord_data,coord_data[:,2:3], forecast_h)
        T = self.T_net(field_features, coord_x, coord_data,coord_data[:,3:4], forecast_h)
        q = self.q_net(field_features, coord_x, coord_data,coord_data[:,4:5], forecast_h)
        rio = self.rio_net(field_features, coord_x, coord_data,coord_data[:,5:6], forecast_h)
        return U, V, P, T, q,rio

    def forward_single(self,variable_name,field_x,coord_x):
        field_features = self.meta_net.forward(field_x)
        x=self.net_dict[variable_name](field_features,coord_x)
        return x





