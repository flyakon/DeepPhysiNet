'''
@Project : deep-downscale 
@File    : variable_net.py
@Author  : Wenyuan Li
@Date    : 2022/7/4 14:34 
@Desc    :  
'''

import torch
import torch.nn  as nn
from DeepPhysiNet.utils.position_encoding import SineCosPE

class ResMLP(nn.Module):
    def __init__(self,in_channels):
        super(ResMLP, self).__init__()
        self.fc=nn.Sequential(nn.Linear(in_channels,in_channels),
                              # nn.Dropout(p=0.5),
                              nn.ReLU(inplace=True),
                              nn.Linear(in_channels, in_channels),
                              # nn.BatchNorm1d(in_channels)
                              )
    def forward(self,x):
        out=self.fc(x)
        return out+x


class VariableNet(nn.Module):
    '''
    inputs: b x token_num x in_channels (d_model)
    '''
    def __init__(self,token_num,in_channels,hidden_channels):
        super(VariableNet, self).__init__()
        self.in_channels=in_channels
        self.hidden_channels=hidden_channels
        self.token_num=token_num
        self.coord_input_fc=nn.Linear(token_num,in_channels+1)
        # self.coord_input_dropout=nn.Dropout(p=0.5)
        self.coord_hidden_fc=nn.Linear(token_num,hidden_channels+1)
        self.data_input_fc=nn.Linear(in_channels,hidden_channels)
        self.fore_h_fc = nn.Linear(in_channels, hidden_channels)
        # self.data_fc=ResMLP(hidden_channels)
        self.cat_fc1=ResMLP(hidden_channels*1)
        # self.cat_fc2 = ResMLP(hidden_channels * 1)
        self.out_fc = nn.Linear(hidden_channels*1,1)
        self.pe=SineCosPE(6,N_freqs=in_channels//2//6,include_input=False)
        self.pe_fore_h = SineCosPE(1, N_freqs=in_channels // 2, include_input=False)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,meta_out:torch.Tensor,coord,coord_data,ref_data,fore_h:torch.Tensor):
        '''

        :param meta_out:  原网络的输出特征，用于计算权重 [1x token_num x in_Cahnenls]
        :param x:         输入的位置和时间参数 [batch x 192]
        :return:
        '''
        batch_size=coord_data.shape[0]
        meta_out=torch.squeeze(meta_out,dim=0) # 312 x 256
        meta_out=meta_out[0:self.token_num]   # 256x 256
        w=self.coord_input_fc.forward(meta_out.T)  # 256 * 193
        w1=w[:,0:self.in_channels]      # 256 x 192
        b1=w[:,self.in_channels]        # 256          # 1

        w=self.coord_hidden_fc.forward(meta_out.T) #256 x 257
        w2 = w[:, 0:self.hidden_channels]  # 256 x 256
        b2 = w[:, self.hidden_channels]     #256

        x=torch.matmul(coord,w1.T) + b1      # b x hidden
        # x = self.coord_input_dropout.forward(x)
        x=self.relu.forward(x)
        x=torch.matmul(x,w2.T) + b2

        # x=self.relu(x)
        coord_data_pe=self.pe.forward(coord_data)
        coord_data_pe=self.data_input_fc.forward(coord_data_pe)
        fore_h=fore_h.squeeze(dim=-1)
        # fore_h = fore_h.repeat((batch_size, 1))
        fore_h_pe = self.pe_fore_h.forward(fore_h)
        fore_h_pe=self.fore_h_fc.forward(fore_h_pe)

        # cat_x=torch.cat((x,coord_data_pe,fore_h_pe),dim=1)
        cat_x=x+coord_data_pe+fore_h_pe
        x=self.cat_fc1.forward(cat_x)
        # x=self.cat_fc2.forward(x)
        x=x+cat_x
        x=self.out_fc.forward(x)
        x=x+ref_data
        return x