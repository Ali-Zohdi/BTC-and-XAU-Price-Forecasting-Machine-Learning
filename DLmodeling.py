from typing import Callable, Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from Transformer_EncDec import Encoder, EncoderLayer, PatchTST_backbone, series_decomp
from SelfAttention_Family import FullAttention, AttentionLayer
from Embed import DataEmbedding_inverted
import numpy as np
import random
import math

######################
## SEGMENTAL MODELS ##
######################

class SegGRU(nn.Module):
    def __init__(self, configs, output_channel):
        super(SegGRU, self).__init__()

        self.output_channel = output_channel

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.gru_enc = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.gru_dec = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)
        self.linear_patch_dr = nn.Linear(self.enc_in, 1)

    def forward(self, x, x_mark, y_mark):

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model

        #### SEGMENT + ENCODING ####
        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)  # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = self.relu(xd)

        enc_out = self.gru_enc(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d

        #### DECODING ####
        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1), # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d

        dec_out = self.gru_dec(dec_in, enc_out)[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1) # B, C, H -> B, H, C

        if self.output_channel == 'single':
            y = self.linear_patch_dr(y).squeeze(2) # B, H, C -> B, H, 1 -> B, H

        y = self.relu(y)

        return y
    
class CustomEmbedingSegGRU(nn.Module):
    def __init__(self, configs, output_channel):
        super(CustomEmbedingSegGRU, self).__init__()

        self.output_channel = output_channel

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.gru_enc = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.gru_dec = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        # self.hour_emb = nn.Parameter(torch.randn(24 * (self.pred_len // self.patch_len), self.d_model // 4))
        # self.day_emb = nn.Parameter(torch.randn(7 * (self.pred_len // self.patch_len), self.d_model // 4))
        self.month_emb = nn.Parameter(torch.randn(12 * (self.pred_len // self.patch_len), self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2)) 

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)
        self.linear_patch_dr = nn.Linear(self.enc_in, 1)

    def forward(self, x, x_mark, y_mark):

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model

        #### SEGMENT + ENCODING ####
        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)  # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = self.relu(xd)

        enc_out = self.gru_enc(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d

        #### DECODING ####
        dec_in = torch.cat([
            # self.hour_emb[(y_mark[..., 3][-1] * 24).type(torch.LongTensor)].unsqueeze(0).repeat(B*C, 1, 1), # M, d//4 -> 1, M, d//4 -> B * C, M, d//4
            # self.day_emb[(y_mark[..., 2][-1] * 7).type(torch.LongTensor)].unsqueeze(0).repeat(B*C, 1, 1), # M, d//4 -> 1, M, d//4 -> B * C, M, d//4
            self.month_emb[(y_mark[..., 1][-1] * 12).type(torch.LongTensor)].unsqueeze(0).repeat(B*C, 1, 1), # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d 

        dec_out = self.gru_dec(dec_in, enc_out)[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1) # B, C, H -> B, H, C
        
        if self.output_channel == 'single':
            y = self.linear_patch_dr(y).squeeze(2) # B, H, C -> B, H, 1 -> B, H

        y = self.relu(y)

        return y

class CNNSegGRU(nn.Module):
    def __init__(self, configs):
        super(CNNSegGRU, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.week_t = configs['week_t']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']
        self.out_channel = configs['out_channel']
        self.cnn_kernel = configs['cnn_kernel']
        self.pooling_size = configs['pooling_size']

        self.conv = configs['conv']

        if self.conv == 1:
            self.conv1 = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )
            
            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel - 1) - 1)//1) + 1
            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Lcnn // self.pooling_size),
                self.seq_len
            )

        if self.conv == 2:
            self.cnn_kernel = (self.week_t, self.patch_len)

            self.conv2 = nn.Conv2d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )

            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Tcnn = ((self.week_t + 2*0 - 1*(self.cnn_kernel[0] - 1) - 1)//1) + 1
            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel[1] - 1) - 1)//1) + 1

            self.flatten = nn.Flatten()

            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Tcnn * self.Lcnn // self.pooling_size),
                self.seq_len
            )            

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.gru_enc = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.gru_dec = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(1, self.d_model // 2))

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x, x_mark, y_mark):

        try:
          B, L, C = x.shape
        except:
          B, T, L, C = x.shape
          
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model
        O = self.out_channel
     # L' = self.Lcnn
     # T' = self.Tcnn
        p = self.pooling_size

        #### CNN ####
        if self.conv == 1:
            x = self.conv1(x.permute(0, 2, 1)) # B, L, C -> B, C, L -> B, O, L'
            x = self.pool(self.bn1(x).relu()) # B, O, L' -> B, O, L'//p
            x = x.view(B, -1) # B, O, L'//p -> B, O * L'//p
            cnn_out = self.linear_patch_cnn(x) # B, O * L'//p -> B, L

        if self.conv == 2:
            x = self.conv2(x.permute(0, 3, 1, 2)) # B, T, L, C -> B, C, T, L -> B, O, T', L'
            x = self.pool(self.bn1(x.view(B, O, -1)).relu()) # B, O, T', L' -> B, O, T'*L' -> B, O, T'*L'//p
            cnn_out = self.linear_patch_cnn(self.flatten(x)) # B, O, T'*L'//p -> B, O * (T'*L'//p) -> B, L            

        #### SEGMENT + ENCODING ####
        seg_w = cnn_out.reshape(B, N, -1)  # B, L -> B, N, W
        seg_d = self.linear_patch(seg_w)  # B, N, W -> B, N, d
        enc_in = self.relu(seg_d)

        enc_out = self.gru_enc(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1, B, d -> 1, B, M * d -> 1, B * M, d

        #### DECONDING ####
        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B, 1, 1), # M, d//2 -> 1, M, d//2 -> B, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # 1, d//2 -> 1, 1, d//2 -> B, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B, M, d -> B * M, d -> B * M, 1, d

        dec_out = self.gru_dec(dec_in, enc_out)[0]  # B * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * M, 1, d -> B * M, 1, W
        y = yw.reshape(B, 1, -1).squeeze(1) # B, 1, H -> B, H
        y = self.relu(y)

        return y    

class CustomEmbedingCNNSegGRU(nn.Module):
    def __init__(self, configs):
        super(CustomEmbedingCNNSegGRU, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.week_t = configs['week_t']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']
        self.out_channel = configs['out_channel']
        self.cnn_kernel = configs['cnn_kernel']
        self.pooling_size = configs['pooling_size']

        self.conv = configs['conv']

        if self.conv == 1:
            self.conv1 = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )
            
            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel - 1) - 1)//1) + 1
            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Lcnn // self.pooling_size),
                self.seq_len
            )

        if self.conv == 2:
            self.cnn_kernel = (self.week_t, self.patch_len)

            self.conv2 = nn.Conv2d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )

            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Tcnn = ((self.week_t + 2*0 - 1*(self.cnn_kernel[0] - 1) - 1)//1) + 1
            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel[1] - 1) - 1)//1) + 1

            self.flatten = nn.Flatten()

            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Tcnn * self.Lcnn // self.pooling_size),
                self.seq_len
            )            

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.gru_enc = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.gru_dec = nn.GRU(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        # self.hour_emb = nn.Parameter(torch.randn(24 * (self.pred_len // self.patch_len), self.d_model // 4))
        # self.day_emb = nn.Parameter(torch.randn(7 * (self.pred_len // self.patch_len), self.d_model // 4))
        self.month_emb = nn.Parameter(torch.randn(12 * (self.pred_len // self.patch_len), self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(1, self.d_model // 2))

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x, x_mark, y_mark):
        # seq_last = x[:, -1:, :].detach()
        # x = x - seq_last
        try:
          B, L, C = x.shape
        except:
          B, T, L, C = x.shape
          
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model
        O = self.out_channel
     # L' = self.Lcnn
     # T' = self.Tcnn
        p = self.pooling_size

        #### CNN ####
        if self.conv == 1:
            x = self.conv1(x.permute(0, 2, 1)) # B, L, C -> B, C, L -> B, O, L'
            x = self.pool(self.bn1(x).relu()) # B, O, L' -> B, O, L'//p
            x = x.view(B, -1) # B, O, L'//p -> B, O * L'//p
            cnn_out = self.linear_patch_cnn(x) # B, O * L'//p -> B, L

        if self.conv == 2:
            x = self.conv2(x.permute(0, 3, 1, 2)) # B, T, L, C -> B, C, T, L -> B, O, T', L'
            x = self.pool(self.bn1(x.view(B, O, -1)).relu()) # B, O, T', L' -> B, O, T'*L' -> B, O, T'*L'//p
            cnn_out = self.linear_patch_cnn(self.flatten(x)) # B, O, T'*L'//p -> B, O * (T'*L'//p) -> B, L            

        #### SEGMENT + ENCODING ####
        seg_w = cnn_out.reshape(B, N, -1)  # B, L -> B, N, W
        seg_d = self.linear_patch(seg_w)  # B, N, W -> B, N, d
        enc_in = self.relu(seg_d)

        enc_out = self.gru_enc(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1, B, d -> 1, B, M * d -> 1, B * M, d

        #### DECONDING ####
        dec_in = torch.cat([
            # self.hour_emb[(y_mark[..., 3][-1] * 24).type(torch.LongTensor)].unsqueeze(0).repeat(B, 1, 1), # M, d//4 -> 1, M, d//4 -> B, M, d//4
            # self.day_emb[(y_mark[..., 2][-1] * 7).type(torch.LongTensor)].unsqueeze(0).repeat(B, 1, 1), # M, d//4 -> 1, M, d//4 -> B, M, d//4
            self.month_emb[(y_mark[..., 1][-1] * 12).type(torch.LongTensor)].unsqueeze(0).repeat(B, 1, 1), # M, d//2 -> 1, M, d//2 -> B, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # 1, d//2 -> 1, 1, d//2 -> B, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d 

        dec_out = self.gru_dec(dec_in, enc_out)[0]  # B * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * M, 1, d -> B * M, 1, W
        y = yw.reshape(B, 1, -1).squeeze(1) # B, 1, H -> B, H
        y = self.relu(y)

        # y = y + seq_last

        return y
    
class SegLSTM(nn.Module):
    def __init__(self, configs, output_channel):
        super(SegLSTM, self).__init__()

        self.output_channel = output_channel

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.lstm_enc = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model, 
            num_layers=1, 
            bias=True, 
            batch_first=True
            )
        self.lstm_dec = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model, 
            num_layers=1, 
            bias=True, 
            batch_first=True
            )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)
        self.linear_patch_dr = nn.Linear(self.enc_in, 1)

    def forward(self, x, x_mark, y_mark):
        # seq_last = x[:, -1:, :].detach()
        # x = x - seq_last

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model

        #### SEGMENT + ENCODING ####
        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)  # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = self.relu(xd)

        _, (enc_out_h, enc_out_c) = self.lstm_enc(enc_in)
        enc_out_h = enc_out_h.repeat(1, 1, M).view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d
        enc_out_c = enc_out_c.repeat(1, 1, M).view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d

        #### DECODING ####
        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1), # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d

        dec_out = self.lstm_dec(dec_in, (enc_out_h, enc_out_c))[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1) # B, C, H -> B, H, C
        
        if self.output_channel == 'single':
            y = self.linear_patch_dr(y).squeeze(2) # B, H, C -> B, H, 1 -> B, H
        
        y = self.relu(y)

        return y
    
class CustomEmbedingSegLSTM(nn.Module):
    def __init__(self, configs, output_channel):
        super(CustomEmbedingSegLSTM, self).__init__()

        self.output_channel = output_channel

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.lstm_enc = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model, 
            num_layers=1, 
            bias=True, 
            batch_first=True
            )
        self.lstm_dec = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model, 
            num_layers=1, 
            bias=True, 
            batch_first=True
            )

        # self.hour_emb = nn.Parameter(torch.randn(24 * (self.pred_len // self.patch_len), self.d_model // 4))
        # self.day_emb = nn.Parameter(torch.randn(7 * (self.pred_len // self.patch_len), self.d_model // 4))
        self.month_emb = nn.Parameter(torch.randn(12 * (self.pred_len // self.patch_len), self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2)) 

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)
        self.linear_patch_dr = nn.Linear(self.enc_in, 1)

    def forward(self, x, x_mark, y_mark):

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model

        #### SEGMENT + ENCODING ####
        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)  # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = self.relu(xd)

        _, (enc_out_h, enc_out_c) = self.lstm_enc(enc_in)
        enc_out_h = enc_out_h.repeat(1, 1, M).view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d
        enc_out_c = enc_out_c.repeat(1, 1, M).view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d

        #### DECODING ####
        dec_in = torch.cat([
            # self.hour_emb[(y_mark[..., 3][-1] * 24).type(torch.LongTensor)].unsqueeze(0).repeat(B*C, 1, 1), # M, d//4 -> 1, M, d//4 -> B * C, M, d//4
            # self.day_emb[(y_mark[..., 2][-1] * 7).type(torch.LongTensor)].unsqueeze(0).repeat(B*C, 1, 1), # M, d//4 -> 1, M, d//4 -> B * C, M, d//4
            self.month_emb[(y_mark[..., 1][-1] * 12).type(torch.LongTensor)].unsqueeze(0).repeat(B*C, 1, 1), # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d 

        dec_out = self.lstm_dec(dec_in, (enc_out_h, enc_out_c))[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1) # B, C, H -> B, H, C
        
        if self.output_channel == 'single':
            y = self.linear_patch_dr(y).squeeze(2) # B, H, C -> B, H, 1 -> B, H
        
        y = self.relu(y)

        return y
    
class CNNSegLSTM(nn.Module):
    def __init__(self, configs):
        super(CNNSegLSTM, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.week_t = configs['week_t']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']
        self.out_channel = configs['out_channel']
        self.cnn_kernel = configs['cnn_kernel']
        self.pooling_size = configs['pooling_size']

        self.conv = configs['conv']

        if self.conv == 1:
            self.conv1 = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )
            
            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel - 1) - 1)//1) + 1
            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Lcnn // self.pooling_size),
                self.seq_len
            )

        if self.conv == 2:
            self.cnn_kernel = (self.week_t, self.patch_len)
            
            self.conv2 = nn.Conv2d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )

            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Tcnn = ((self.week_t + 2*0 - 1*(self.cnn_kernel[0] - 1) - 1)//1) + 1
            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel[1] - 1) - 1)//1) + 1

            self.flatten = nn.Flatten()

            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Tcnn * self.Lcnn // self.pooling_size),
                self.seq_len
            )

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.lstm_enc = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model, 
            num_layers=1, 
            bias=True, 
            batch_first=True
            )
        self.lstm_dec = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model, 
            num_layers=1, 
            bias=True, 
            batch_first=True
            )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(1, self.d_model // 2))

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x, x_mark, y_mark):

        try:
          B, L, C = x.shape
        except:
          B, T, L, C = x.shape

        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model
        O = self.out_channel
     # L' = self.Lcnn
     # T' = self.Tcnn
        p = self.pooling_size

        #### CNN ####
        if self.conv == 1:
            x = self.conv1(x.permute(0, 2, 1)) # B, L, C -> B, C, L -> B, O, L'
            x = self.pool(self.bn1(x).relu()) # B, O, L' -> B, O, L'//p
            x = x.view(B, -1) # B, O, L'//p -> B, O * L'//p
            cnn_out = self.linear_patch_cnn(x) # B, O * L'//p -> B, L

        if self.conv == 2:
            x = self.conv2(x.permute(0, 3, 1, 2)) # B, T, L, C -> B, C, T, L -> B, O, T', L'
            x = self.pool(self.bn1(x.view(B, O, -1)).relu()) # B, O, T', L' -> B, O, T'*L' -> B, O, T'*L'//p
            cnn_out = self.linear_patch_cnn(x.view(B, -1)) # B, O, T'*L'//p -> B, O * (T'*L'//p) -> B, L  

        #### SEGMENT + ENCODING ####
        seg_w = cnn_out.reshape(B, N, -1)  # B, L -> B, N, W
        seg_d = self.linear_patch(seg_w)  # B, N, W -> B, N, d
        enc_in = self.relu(seg_d)

        _, (enc_out_h, enc_out_c) = self.lstm_enc(enc_in)
        enc_out_h = enc_out_h.repeat(1, 1, M).view(1, -1, self.d_model) # 1, B, d -> 1, B, M * d -> 1, B * M, d
        enc_out_c = enc_out_c.repeat(1, 1, M).view(1, -1, self.d_model) # 1, B, d -> 1, B, M * d -> 1, B * M, d

       #### DECODING ####
        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B, 1, 1), # M, d//2 -> 1, M, d//2 -> B, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # 1, d//2 -> 1, 1, d//2 -> B, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B, M, d -> B * M, d -> B * M, 1, d

        dec_out = self.lstm_dec(dec_in, (enc_out_h, enc_out_c))[0]  # B * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * M, 1, d -> B * M, 1, W
        y = yw.reshape(B, 1, -1).squeeze(1) # B, 1, H -> B, H
        y = self.relu(y)

        return y
    
class CustomEmbedingCNNSegLSTM(nn.Module):
    def __init__(self, configs):
        super(CustomEmbedingCNNSegLSTM, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.week_t = configs['week_t']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']
        self.out_channel = configs['out_channel']
        self.cnn_kernel = configs['cnn_kernel']
        self.pooling_size = configs['pooling_size']

        self.conv = configs['conv']

        if self.conv == 1:
            self.conv1 = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )
            
            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel - 1) - 1)//1) + 1
            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Lcnn // self.pooling_size),
                self.seq_len
            )

        if self.conv == 2:
            self.cnn_kernel = (self.week_t, self.patch_len)
            
            self.conv2 = nn.Conv2d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )

            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Tcnn = ((self.week_t + 2*0 - 1*(self.cnn_kernel[0] - 1) - 1)//1) + 1
            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel[1] - 1) - 1)//1) + 1

            self.flatten = nn.Flatten()

            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Tcnn * self.Lcnn // self.pooling_size),
                self.seq_len
            )

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.lstm_enc = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model, 
            num_layers=1, 
            bias=True, 
            batch_first=True
            )
        self.lstm_dec = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model, 
            num_layers=1, 
            bias=True, 
            batch_first=True
            )

        # self.hour_emb = nn.Parameter(torch.randn(24 * (self.pred_len // self.patch_len), self.d_model // 2))
        # self.day_emb = nn.Parameter(torch.randn(7 * (self.pred_len // self.patch_len), self.d_model // 2))
        self.month_emb = nn.Parameter(torch.randn(12 * (self.pred_len // self.patch_len), self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(1, self.d_model // 2)) 

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x, x_mark, y_mark):
        
        try:
          B, L, C = x.shape
        except:
          B, T, L, C = x.shape

        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model
        O = self.out_channel
     # L' = self.Lcnn
     # T' = self.Tcnn
        p = self.pooling_size

        #### CNN ####
        if self.conv == 1:
            x = self.conv1(x.permute(0, 2, 1)) # B, L, C -> B, C, L -> B, O, L'
            x = self.pool(self.bn1(x).relu()) # B, O, L' -> B, O, L'//p
            x = x.view(B, -1) # B, O, L'//p -> B, O * L'//p
            cnn_out = self.linear_patch_cnn(x) # B, O * L'//p -> B, L

        if self.conv == 2:
            x = self.conv2(x.permute(0, 3, 1, 2)) # B, T, L, C -> B, C, T, L -> B, O, T', L'
            x = self.pool(self.bn1(x.view(B, O, -1)).relu()) # B, O, T', L' -> B, O, T'*L' -> B, O, T'*L'//p
            cnn_out = self.linear_patch_cnn(self.flatten(x)) # B, O, T'*L'//p -> B, O * (T'*L'//p) -> B, L  

        #### SEGMENT + ENCODING ####
        seg_w = cnn_out.reshape(B, N, -1)  # B, L -> B, N, W
        seg_d = self.linear_patch(seg_w)  # B, N, W -> B, N, d
        enc_in = self.relu(seg_d)

        _, (enc_out_h, enc_out_c) = self.lstm_enc(enc_in)
        enc_out_h = enc_out_h.repeat(1, 1, M).view(1, -1, self.d_model) # 1, B, d -> 1, B, M * d -> 1, B * M, d
        enc_out_c = enc_out_c.repeat(1, 1, M).view(1, -1, self.d_model) # 1, B, d -> 1, B, M * d -> 1, B * M, d

       #### DECODING ####
        dec_in = torch.cat([
            self.hour_emb[(y_mark[..., 3][-1] * 24).type(torch.LongTensor)].unsqueeze(0).repeat(B, M, 1), # 1, d//4 -> 1, 1, d//4 -> B, M, d//4
            self.day_emb[(y_mark[..., 2][-1] * 7).type(torch.LongTensor)].unsqueeze(0).repeat(B, M, 1), # 1, d//4 -> 1, 1, d//4 -> B, M, d//4
            self.month_emb[(y_mark[..., 1][-1] * 12).type(torch.LongTensor)].unsqueeze(0).repeat(B, M, 1), # 1, d//4 -> 1, 1, d//4 -> B, M, d//4
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # 1, d//4 -> 1, 1, d//4 -> B, M, d//4
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d 

        dec_out = self.lstm_dec(dec_in, (enc_out_h, enc_out_c))[0]  # B * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * M, 1, d -> B * M, 1, W
        y = yw.reshape(B, 1, -1).squeeze(1) # B, 1, H -> B, H
        y = self.relu(y)
        return y

class SegRNN(nn.Module):
    def __init__(self, configs, output_channel):
        super(SegRNN, self).__init__()

        self.output_channel = output_channel

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.rnn_enc = nn.RNN(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            num_layers=1, 
            nonlinearity='tanh', 
            bias=True, 
            batch_first=True
            )
        self.rnn_dec = nn.RNN(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            num_layers=1, 
            nonlinearity='tanh', 
            bias=True, 
            batch_first=True
            )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)
        self.linear_patch_dr = nn.Linear(self.enc_in, 1)

    def forward(self, x, x_mark, y_mark):

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model

        #### SEGMENT + ENCODING ####
        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)  # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = self.relu(xd)

        enc_out = self.rnn_enc(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d

        #### DECODING ####
        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1), # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d

        dec_out = self.rnn_dec(dec_in, enc_out)[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1) # B, C, H -> B, H, C
        
        if self.output_channel == 'single':
            y = self.linear_patch_dr(y).squeeze(2) # B, H, C -> B, H, 1 -> B, H
    
        y = self.relu(y)

        return y
    
class CustomEmbedingSegRNN(nn.Module):
    def __init__(self, configs, output_channel):
        super(CustomEmbedingSegRNN, self).__init__()

        self.output_channel = output_channel

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']
        
        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.rnn_enc = nn.RNN(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            num_layers=1, 
            nonlinearity='tanh', 
            bias=True, 
            batch_first=True
            )
        self.rnn_dec = nn.RNN(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            num_layers=1, 
            nonlinearity='tanh', 
            bias=True, 
            batch_first=True
            )

        # self.hour_emb = nn.Parameter(torch.randn(24 * (self.pred_len // self.patch_len), self.d_model // 4))
        # self.day_emb = nn.Parameter(torch.randn(7 * (self.pred_len // self.patch_len), self.d_model // 4))
        self.month_emb = nn.Parameter(torch.randn(12 * (self.pred_len // self.patch_len), self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))                

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)
        self.linear_patch_dr = nn.Linear(self.enc_in, 1)

    def forward(self, x, x_mark, y_mark):

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model

        #### SEGMENT + ENCODING ####
        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)  # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = self.relu(xd)

        enc_out = self.rnn_enc(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d

        #### DECODING ####
        dec_in = torch.cat([
            # self.hour_emb[(y_mark[..., 3][-1] * 24).type(torch.LongTensor)].unsqueeze(0).repeat(B*C, 1, 1), # M, d//4 -> 1, M, d//4 -> B * C, M, d//4
            # self.day_emb[(y_mark[..., 2][-1] * 7).type(torch.LongTensor)].unsqueeze(0).repeat(B*C, 1, 1), # M, d//4 -> 1, M, d//4 -> B * C, M, d//4
            self.month_emb[(y_mark[..., 1][-1] * 12).type(torch.LongTensor)].unsqueeze(0).repeat(B*C, 1, 1), # M, d//2 -> 1, M, d//2 -> B * C, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # C, d//2 -> C, 1, d//2 -> B * C, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d 

        dec_out = self.rnn_dec(dec_in, enc_out)[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1) # B, C, H -> B, H, C

        if self.output_channel == 'single':
            y = self.linear_patch_dr(y).squeeze(2) # B, H, C -> B, H, 1 -> B, H
    
        y = self.relu(y)

        return y           
    
class CNNSegRNN(nn.Module):
    def __init__(self, configs):
        super(CNNSegRNN, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.week_t = configs['week_t']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']
        self.out_channel = configs['out_channel']
        self.cnn_kernel = configs['cnn_kernel']
        self.pooling_size = configs['pooling_size']

        self.conv = configs['conv']

        if self.conv == 1:
            self.conv1 = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )
            
            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel - 1) - 1)//1) + 1
            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Lcnn // self.pooling_size),
                self.seq_len
            )

        if self.conv == 2:
            self.cnn_kernel = (self.week_t, self.patch_len)
            
            self.conv2 = nn.Conv2d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )

            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Tcnn = ((self.week_t + 2*0 - 1*(self.cnn_kernel[0] - 1) - 1)//1) + 1
            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel[1] - 1) - 1)//1) + 1

            self.flatten = nn.Flatten()

            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Tcnn * self.Lcnn // self.pooling_size),
                self.seq_len
            )

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.rnn_enc = nn.RNN(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            num_layers=1, 
            nonlinearity='tanh', 
            bias=True, 
            batch_first=True
            )
        self.rnn_dec = nn.RNN(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            num_layers=1, 
            nonlinearity='tanh', 
            bias=True, 
            batch_first=True
            )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(1, self.d_model // 2))

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x, x_mark, y_mark):

        try:
          B, L, C = x.shape
        except:
          B, T, L, C = x.shape

        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model
        O = self.out_channel
     # L' = self.Lcnn
     # T' = self.Tcnn
        p = self.pooling_size

        #### CNN ####
        if self.conv == 1:
            x = self.conv1(x.permute(0, 2, 1)) # B, L, C -> B, C, L -> B, O, L'
            x = self.pool(self.bn1(x).relu()) # B, O, L' -> B, O, L'//p
            x = x.view(B, -1) # B, O, L'//p -> B, O * L'//p
            cnn_out = self.linear_patch_cnn(x) # B, O * L'//p -> B, L

        if self.conv == 2:
            x = self.conv2(x.permute(0, 3, 1, 2)) # B, T, L, C -> B, C, T, L -> B, O, T', L'
            x = self.pool(self.bn1(x.view(B, O, -1)).relu()) # B, O, T', L' -> B, O, T'*L' -> B, O, T'*L'//p
            cnn_out = self.linear_patch_cnn(self.flatten(x)) # B, O, T'*L'//p -> B, O * (T'*L'//p) -> B, L 

        #### SEGMENT + ENCODING ####
        seg_w = cnn_out.reshape(B, N, -1)  # B, L -> B, N, W
        seg_d = self.linear_patch(seg_w)  # B, N, W -> B, N, d
        enc_in = self.relu(seg_d)

        enc_out = self.rnn_enc(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1, B, d -> 1, B, M * d -> 1, B * M, d

        #### DECONDING ####
        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B, 1, 1), # M, d//2 -> 1, M, d//2 -> B, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # 1, d//2 -> 1, 1, d//2 -> B, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B, M, d -> B * M, d -> B * M, 1, d

        dec_out = self.rnn_dec(dec_in, enc_out)[0]  # B * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * M, 1, d -> B * M, 1, W
        y = yw.reshape(B, 1, -1).squeeze(1) # B, 1, H -> B, H
        y = self.relu(y)

        # y = y + seq_last

        return y

class CustomEmbedingCNNSegRNN(nn.Module):
    def __init__(self, configs):
        super(CustomEmbedingCNNSegRNN, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.week_t = configs['week_t']
        self.enc_in = configs['enc_in']
        self.patch_len = configs['patch_len']
        self.d_model = configs['d_model']
        self.out_channel = configs['out_channel']
        self.cnn_kernel = configs['cnn_kernel']
        self.pooling_size = configs['pooling_size']

        self.conv = configs['conv']

        if self.conv == 1:
            self.conv1 = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )
            
            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel - 1) - 1)//1) + 1
            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Lcnn // self.pooling_size),
                self.seq_len
            )

        if self.conv == 2:
            self.cnn_kernel = (self.week_t, self.patch_len)
            
            self.conv2 = nn.Conv2d(
                in_channels=self.enc_in,
                out_channels=self.out_channel,
                kernel_size=self.cnn_kernel
                )

            self.bn1 = nn.BatchNorm1d(self.out_channel)
            self.pool = nn.MaxPool1d(self.pooling_size)

            self.Tcnn = ((self.week_t + 2*0 - 1*(self.cnn_kernel[0] - 1) - 1)//1) + 1
            self.Lcnn = ((self.seq_len + 2*0 - 1*(self.cnn_kernel[1] - 1) - 1)//1) + 1

            self.flatten = nn.Flatten()

            self.linear_patch_cnn = nn.Linear(
                self.out_channel * (self.Tcnn * self.Lcnn // self.pooling_size),
                self.seq_len
            )

        self.linear_patch = nn.Linear(self.patch_len, self.d_model)
        self.relu = nn.ReLU()

        self.rnn_enc = nn.RNN(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            num_layers=1, 
            nonlinearity='tanh', 
            bias=True, 
            batch_first=True
            )
        self.rnn_dec = nn.RNN(
            input_size=self.d_model, 
            hidden_size=self.d_model, 
            num_layers=1, 
            nonlinearity='tanh', 
            bias=True, 
            batch_first=True
            )

        # self.hour_emb = nn.Parameter(torch.randn(24 * (self.pred_len // self.patch_len), self.d_model // 4))
        # self.day_emb = nn.Parameter(torch.randn(7 * (self.pred_len // self.patch_len), self.d_model // 4))
        self.month_emb = nn.Parameter(torch.randn(12 * (self.pred_len // self.patch_len), self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(1, self.d_model // 2))  

        self.dropout = nn.Dropout(configs['dropout'])
        self.linear_patch_re = nn.Linear(self.d_model, self.patch_len)

    def forward(self, x, x_mark, y_mark):

        try:
          B, L, C = x.shape
        except:
          B, T, L, C = x.shape

        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.d_model
        O = self.out_channel
     # L' = self.Lcnn
     # T' = self.Tcnn
        p = self.pooling_size

        #### CNN ####
        if self.conv == 1:
            x = self.conv1(x.permute(0, 2, 1)) # B, L, C -> B, C, L -> B, O, L'
            x = self.pool(self.bn1(x).relu()) # B, O, L' -> B, O, L'//p
            x = x.view(B, -1) # B, O, L'//p -> B, O * L'//p
            cnn_out = self.linear_patch_cnn(x) # B, O * L'//p -> B, L

        if self.conv == 2:
            x = self.conv2(x.permute(0, 3, 1, 2)) # B, T, L, C -> B, C, T, L -> B, O, T', L'
            x = self.pool(self.bn1(x.view(B, O, -1)).relu()) # B, O, T', L' -> B, O, T'*L' -> B, O, T'*L'//p
            cnn_out = self.linear_patch_cnn(self.flatten(x)) # B, O, T'*L'//p -> B, O * (T'*L'//p) -> B, L 

        #### SEGMENT + ENCODING ####
        seg_w = cnn_out.reshape(B, N, -1)  # B, L -> B, N, W
        seg_d = self.linear_patch(seg_w)  # B, N, W -> B, N, d
        enc_in = self.relu(seg_d)

        enc_out = self.rnn_enc(enc_in)[1].repeat(1, 1, M).view(1, -1, self.d_model) # 1, B, d -> 1, B, M * d -> 1, B * M, d

        #### DECONDING ####
        dec_in = torch.cat([
            # self.hour_emb[(y_mark[..., 3][-1] * 24).type(torch.LongTensor)].unsqueeze(0).repeat(B, 1, 1), # M, d//4 -> 1, M, d//4 -> B, M, d//4
            # self.day_emb[(y_mark[..., 2][-1] * 7).type(torch.LongTensor)].unsqueeze(0).repeat(B, 1, 1), # M, d//4 -> 1, M, d//4 -> B, M, d//4
            self.month_emb[(y_mark[..., 1][-1] * 12).type(torch.LongTensor)].unsqueeze(0).repeat(B, 1, 1), # M, d//2 -> 1, M, d//2 -> B, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # 1, d//2 -> 1, 1, d//2 -> B, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B * C, M, d -> B * C * M, d -> B * C * M, 1, d 

        dec_out = self.rnn_dec(dec_in, enc_out)[0]  # B * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * M, 1, d -> B * M, 1, W
        y = yw.reshape(B, 1, -1).squeeze(1) # B, 1, H -> B, H
        y = self.relu(y)

        return y
    
#######################
## SEQUENTIAL MODELS ##
#######################

class EncoderRNN(nn.Module):
    def __init__(self, configs):
        super(EncoderRNN, self).__init__()

        self.enc_in = configs['enc_in']
        self.d_model = configs['d_model']

        self.rnn = nn.RNN(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=1, 
            nonlinearity='tanh', 
            bias=True, 
            batch_first=True
            )


    def forward(self, x):

        B, L, C = x.shape
        d = self.d_model

        enc_out = self.rnn(x)[1] # B, L, C -> 1, B, d

        return enc_out


class DecoderRNN(nn.Module):
    def __init__(self, configs):
        super(DecoderRNN, self).__init__()

        self.seq_len = configs['seq_len']
        self.d_model = configs['d_model']
        self.de_pred_len = configs['patch_len']

        self.rnn = nn.RNN(
            input_size=self.seq_len,
            hidden_size=self.d_model,
            num_layers=1, 
            nonlinearity='tanh', 
            bias=True, 
            batch_first=True
            )


        self.relu = nn.ReLU()
        self.linear_patch = nn.Linear(self.d_model, self.de_pred_len)

    def forward(self, input, hidden_state):

        B, L, a = input.shape # a = 1
        a, B, d = hidden_state.shape # a = 1 as long as RNN layers are 1
        W = self.de_pred_len

        input = input.permute(0, 2, 1) # B, L, 1 -> B, 1, L
        dec_out, hid= self.rnn(input, hidden_state) # (B, 1, L)&(1, B, d) -> (B, 1, d)&(1, B, d)

        de_pred = self.linear_patch(dec_out) # B, 1, d -> B, 1, W
        de_pred = de_pred.view(B, -1) # B, W
        de_pred = self.relu(de_pred)

        return de_pred, hid
    
class Seq2SeqRNN(nn.Module):
    def __init__(self, configs):
        super(Seq2SeqRNN, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.patch_len = configs['patch_len']

        self.encoder = EncoderRNN(configs)
        self.decoder = DecoderRNN(configs)

    def forward(self, x, y, teacher_forcing_ratio):

        B, L, C = x.shape
        H = self.pred_len
        W = self.patch_len
        M = self.pred_len//self.patch_len
        
        outputs = torch.zeros(B, H)
        hidden_state = self.encoder(x) # B, L, C -> 1, B, d

        x_input = x[:, :, :1] # B, L, 1 - choosing the column that we are predicting the value

        for m in range(M):

            output, hidden_state = self.decoder(x_input, hidden_state) # (B, L, 1)&(1, B, d) -> (B, W)&(1, B, d)

            outputs[:, m*self.patch_len:(m+1)*self.patch_len] = output # add the output to the m-th section of the outputs

            if random.random() < teacher_forcing_ratio:
                appending = y[:, m*self.patch_len:(m+1)*self.patch_len]
               
            else:
                appending = output
            
            appending = appending.unsqueeze(2) # B, W -> B, W, 1

            x_input = x_input[:, self.patch_len:, :1] # B, L, 1 -> B, L - W, 1

            x_input = torch.cat((x_input, appending), dim=1) # B, L - W, 1 -> B, L, 1

        return outputs

class EncoderGRU(nn.Module):
    def __init__(self, configs):
        super(EncoderGRU, self).__init__()

        self.enc_in = configs['enc_in']
        self.d_model = configs['d_model']

        self.gru = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

    def forward(self, x):

        B, L, C = x.shape
        d = self.d_model

        enc_out = self.gru(x)[1] # B, L, C -> 1, B, d

        return enc_out


class DecoderGRU(nn.Module):
    def __init__(self, configs):
        super(DecoderGRU, self).__init__()

        self.seq_len = configs['seq_len']
        self.d_model = configs['d_model']
        self.de_pred_len = configs['patch_len']

        self.gru = nn.GRU(
            input_size=self.seq_len,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True
        )

        self.relu = nn.ReLU()
        self.linear_patch = nn.Linear(self.d_model, self.de_pred_len)

    def forward(self, input, hidden_state):

        B, L, a = input.shape # a = 1
        a, B, d = hidden_state.shape # a = 1 as long as RNN layers are 1
        W = self.de_pred_len

        input = input.permute(0, 2, 1) # B, L, 1 -> B, 1, L
        dec_out, hid= self.gru(input, hidden_state) # (B, 1, L)&(1, B, d) -> (B, 1, d)&(1, B, d)

        de_pred = self.linear_patch(dec_out) # B, 1, d -> B, 1, W
        de_pred = de_pred.view(B, -1) # B, W
        de_pred = self.relu(de_pred)

        return de_pred, hid
    
class Seq2SeqGRU(nn.Module):
    def __init__(self, configs):
        super(Seq2SeqGRU, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.patch_len = configs['patch_len']

        self.encoder = EncoderGRU(configs)
        self.decoder = DecoderGRU(configs)

    def forward(self, x, y, teacher_forcing_ratio):

        B, L, C = x.shape
        H = self.pred_len
        W = self.patch_len
        M = self.pred_len//self.patch_len
        
        outputs = torch.zeros(B, H)
        hidden_state = self.encoder(x) # B, L, C -> 1, B, d

        x_input = x[:, :, :1] # B, L, 1 - choosing the column that we are predicting the value

        for m in range(M):

            output, hidden_state = self.decoder(x_input, hidden_state) # (B, L, 1)&(1, B, d) -> (B, W)&(1, B, d)

            outputs[:, m*self.patch_len:(m+1)*self.patch_len] = output # add the output to the m-th section of the outputs

            if random.random() < teacher_forcing_ratio:
                appending = y[:, m*self.patch_len:(m+1)*self.patch_len]
               
            else:
                appending = output
            
            appending = appending.unsqueeze(2) # B, W -> B, W, 1

            x_input = x_input[:, self.patch_len:, :1] # B, L, 1 -> B, L - W, 1

            x_input = torch.cat((x_input, appending), dim=1) # B, L - W, 1 -> B, L, 1

        return outputs
    
class EncoderLSTM(nn.Module):
    def __init__(self, configs):
        super(EncoderLSTM, self).__init__()

        self.enc_in = configs['enc_in']
        self.d_model = configs['d_model']

        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

    def forward(self, x):

        B, L, C = x.shape
        d = self.d_model

        enc_out = self.lstm(x)[1] # B, L, C -> 1, B, d

        return enc_out


class DecoderLSTM(nn.Module):
    def __init__(self, configs):
        super(DecoderLSTM, self).__init__()

        self.seq_len = configs['seq_len']
        self.d_model = configs['d_model']
        self.de_pred_len = configs['patch_len']

        self.lstm = nn.LSTM(
            input_size=self.seq_len,
            hidden_size=self.d_model,
            num_layers=1,
            bias=True,
            batch_first=True
        )

        self.relu = nn.ReLU()
        self.linear_patch = nn.Linear(self.d_model, self.de_pred_len)

    def forward(self, input, hidden_state):

        B, L, a = input.shape # a = 1
        a, B, d = hidden_state[0].shape # a = 1 as long as RNN layers are 1
        W = self.de_pred_len

        input = input.permute(0, 2, 1) # B, L, 1 -> B, 1, L
        dec_out, hid= self.lstm(input, hidden_state) # (B, 1, L)&(1, B, d) -> (B, 1, d)&(1, B, d)

        de_pred = self.linear_patch(dec_out) # B, 1, d -> B, 1, W
        de_pred = de_pred.view(B, -1) # B, W
        de_pred = self.relu(de_pred)

        return de_pred, hid
    
class Seq2SeqLSTM(nn.Module):
    def __init__(self, configs):
        super(Seq2SeqLSTM, self).__init__()

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.patch_len = configs['patch_len']

        self.encoder = EncoderLSTM(configs)
        self.decoder = DecoderLSTM(configs)

    def forward(self, x, y, teacher_forcing_ratio):

        B, L, C = x.shape
        H = self.pred_len
        W = self.patch_len
        M = self.pred_len//self.patch_len
        
        outputs = torch.zeros(B, H)
        hidden_state = self.encoder(x) # B, L, C -> 1, B, d

        x_input = x[:, :, :1] # B, L, 1 - choosing the column that we are predicting the value

        for m in range(M):

            output, hidden_state = self.decoder(x_input, hidden_state) # (B, L, 1)&(1, B, d) -> (B, W)&(1, B, d)

            outputs[:, m*self.patch_len:(m+1)*self.patch_len] = output # add the output to the m-th section of the outputs

            if random.random() < teacher_forcing_ratio:
                appending = y[:, m*self.patch_len:(m+1)*self.patch_len]
               
            else:
                appending = output
            
            appending = appending.unsqueeze(2) # B, W -> B, W, 1

            x_input = x_input[:, self.patch_len:, :1] # B, L, 1 -> B, L - W, 1

            x_input = torch.cat((x_input, appending), dim=1) # B, L - W, 1 -> B, L, 1

        return outputs
    
########################
## TRANSFORMER MODELS ##
########################
    
class iTransformer(nn.Module):

    def __init__(self, configs, output_channel):
        super(iTransformer, self).__init__()

        self.output_channel = output_channel

        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        self.d_model = configs['d_model']
        self.enc_in = configs['enc_in']
        self.d_ff = configs['d_ff']
        self.output_attention = configs['output_attention']
        self.use_norm = configs['use_norm']
        self.dropout = configs['dropout']
        self.n_heads = configs['n_heads']
        self.e_layers = configs['e_layers']

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, 'fixed', 'h',
                                                    self.dropout)
        
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag=False, factor=5, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout
                    # activation=configs.activation # -> activation is relu
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)
        self.linear_patch_dr = nn.Linear(self.enc_in, 1)

        self.relu = nn.ReLU()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # if self.use_norm:
        #     # Normalization from Non-stationary Transformer
        #     means = x_enc.mean(1, keepdim=True).detach()
        #     x_enc = x_enc - means
        #     stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        #     x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.output_channel == 'single':
            # B S N -> B S 1 -> B S
            dec_out = self.linear_patch_dr(dec_out)
       
        dec_out = self.relu(dec_out)

        # if self.use_norm:
        #     # De-Normalization from Non-stationary Transformer
        #     dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        #     dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        dec_out = dec_out[:, -self.pred_len:, :]
        
        if self.output_channel == 'single':
            dec_out = dec_out.squeeze(2)

        return dec_out  # [B, L, D]
    
class PatchTST(nn.Module):
    def __init__(self, configs, output_channel, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, verbose:bool=False, **kwargs):
        
        super(PatchTST, self).__init__()

        max_seq_len=1024
        self.head_type = output_channel
        
        # load parameters
        c_in = configs['enc_in']
        context_window = configs['seq_len']
        target_window = configs['pred_len']
        
        n_layers = configs['e_layers']
        n_heads = configs['n_heads']
        d_model = configs['d_model']
        d_ff = configs['d_ff']
        dropout = configs['dropout']
        fc_dropout = configs['fc_dropout']
        head_dropout = configs['head_dropout']
        
        individual = configs['individual']
    
        patch_len = configs['patch_len']
        stride = configs['stride']
        padding_patch = configs['padding_patch']
        
        revin = False # configs['revin']
        affine = configs['affine']
        subtract_last = configs['subtract_last']
        
        decomposition = configs['decomposition']
        kernel_size = configs['kernel_size']
        
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=self.head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=self.head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=self.head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            if self.head_type == 'multi':
                x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            if self.head_type == 'multi':
                x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x
    
configs = {
    'enc_in' : 4,
    'seq_len' : 168,
    'pred_len' : 24,
    'e_layers' : 1,
    'n_heads' : 4,
    'd_model' : 256,
    'd_ff' : 1024,
    'dropout' : 0.5,
    'fc_dropout' : 0.2,
    'head_dropout' : 0.1,
    'individual' : True,
    'patch_len' : 6,
    'stride' : 6,
    'padding_patch' : 'no', # its default value is 'end' and it generates ReplicationPad1d padding
    'affine' : False, # used for RevIN normalization which is eliminated
    'subtract_last' : False, # used for RevIN normalization which is eliminated 
    'decomposition' : True,
    'kernel_size' : 3 # time series decomposition kernel

}

inputs = torch.randn((2000, 168, 4))

model = PatchTST(configs=configs, output_channel='single')

outputs = model(inputs)

print(outputs.shape)