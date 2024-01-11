'''
@Project : deep-downscale 
@File    : transformer_net.py
@Author  : Wenyuan Li
@Date    : 2022/2/12 14:31 
@Desc    :  some codes are reffered from Informer: https://github.com/zhouhaoyi/Informer2020.git
'''

import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from .embed import DataEmbedding
from .attn import ProbAttention,FullAttention,AttentionLayer

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + new_x

        y = x = self.norm1(x)
        y = self.activation(self.conv1(y.transpose(-1, 1)))
        y = self.conv2(y).transpose(-1, 1)

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns


class TransformerNet(nn.Module):
    def __init__(self, enc_in, c_out,
                d_model=512, n_heads=8, e_layers=6, d_ff=512,
                 activation='gelu',learnable_token_num=128,
                 output_attention=False,**kwargs):
        super(TransformerNet, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model)
        token=torch.rand([1,learnable_token_num,d_model])
        self.learnable_token=nn.Parameter(token,requires_grad=True)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc,forecast_h,
                enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc,forecast_h,self.learnable_token)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        enc_out = self.projection(enc_out)
        return enc_out





