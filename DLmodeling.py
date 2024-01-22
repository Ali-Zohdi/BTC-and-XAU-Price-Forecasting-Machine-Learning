import torch
import torch.nn as nn
import math

class SegGRU(nn.Module):
    def __init__(self, configs):
        super(SegGRU, self).__init__()

        # remove this, the performance will be bad
        # self.lucky = nn.Embedding(configs.enc_in, configs.d_model // 2)

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
        y = self.linear_patch_dr(y).squeeze(2) # B, H, C -> B, H, 1 -> B, H
        y = self.relu(y)

        # y = y + seq_last

        return y

class CNNSegGRU(nn.Module):
    def __init__(self, configs):
        super(CNNSegGRU, self).__init__()

        # remove this, the performance will be bad
        # self.lucky = nn.Embedding(configs.enc_in, configs.d_model // 2)

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
            self.Lcnn = ((self.pred_len + 2*0 - 1*(self.cnn_kernel[1] - 1) - 1)//1) + 1

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
            self.pos_emb.unsqueeze(0).repeat(B, 1, 1), # M, d//2 -> 1, M, d//2 -> B, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # 1, d//2 -> 1, 1, d//2 -> B, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B, M, d -> B * M, d -> B * M, 1, d

        dec_out = self.gru_dec(dec_in, enc_out)[0]  # B * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * M, 1, d -> B * M, 1, W
        y = yw.reshape(B, 1, -1).squeeze(1) # B, 1, H -> B, H
        y = self.relu(y)

        # y = y + seq_last

        return y    

class SegLSTM(nn.Module):
    def __init__(self, configs):
        super(SegLSTM, self).__init__()

        # remove this, the performance will be bad
        # self.lucky = nn.Embedding(configs.enc_in, configs.d_model // 2)

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
        y = self.linear_patch_dr(y).squeeze(2) # B, H, C -> B, H, 1 -> B, H
        y = self.relu(y)

        # y = y + seq_last

        return y
    
class CNNSegLSTM(nn.Module):
    def __init__(self, configs):
        super(CNNSegLSTM, self).__init__()

        # remove this, the performance will be bad
        # self.lucky = nn.Embedding(configs.enc_in, configs.d_model // 2)

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
            self.Lcnn = ((self.pred_len + 2*0 - 1*(self.cnn_kernel[1] - 1) - 1)//1) + 1

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

        _, (enc_out_h, enc_out_c) = self.lstm_enc(enc_in)
        enc_out_h = enc_out_h.repeat(1, 1, M).view(1, -1, self.d_model) # 1, B, d -> 1, B, M * d -> 1, B * M, d
        enc_out_c = enc_out_c.repeat(1, 1, M).view(1, -1, self.d_model) # 1, B, d -> 1, B, M * d -> 1, B * M, d

        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B, 1, 1), # M, d//2 -> 1, M, d//2 -> B, M, d//2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1) # 1, d//2 -> 1, 1, d//2 -> B, M, d//2
        ], dim=-1).flatten(0, 1).unsqueeze(1) # B, M, d -> B * M, d -> B * M, 1, d

        dec_out = self.lstm_dec(dec_in, (enc_out_h, enc_out_c))[0]  # B * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * M, 1, d -> B * M, 1, W
        y = yw.reshape(B, 1, -1).squeeze(1) # B, 1, H -> B, H
        y = self.relu(y)

        # y = y + seq_last

        return y

class SegRNN(nn.Module):
    def __init__(self, configs):
        super(SegRNN, self).__init__()

        # remove this, the performance will be bad
        # self.lucky = nn.Embedding(configs.enc_in, configs.d_model // 2)

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
        y = self.linear_patch_dr(y).squeeze(2) # B, H, C -> B, H, 1 -> B, H
        y = self.relu(y)

        # y = y + seq_last

        return y
    
class CNNSegRNN(nn.Module):
    def __init__(self, configs):
        super(CNNSegRNN, self).__init__()

        # remove this, the performance will be bad
        # self.lucky = nn.Embedding(configs.enc_in, configs.d_model // 2)

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
            self.Lcnn = ((self.pred_len + 2*0 - 1*(self.cnn_kernel[1] - 1) - 1)//1) + 1

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