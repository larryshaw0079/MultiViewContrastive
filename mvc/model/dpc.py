"""
@Time    : 2020/11/10 21:27
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : dpc.py
@Software: PyCharm
@Desc    : 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import R1DNet, R2DNet, ConvGRU1d, ConvGRU2d


class DPC(nn.Module):
    def __init__(self, network, input_channels, hidden_channels, feature_dim, pred_steps, temperature, device):
        super(DPC, self).__init__()

        self.network = network
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.temperature = temperature
        self.device = device

        if network == 'r1d':
            self.encoder = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                                  final_fc=False)
            feature_size = self.encoder.feature_size
            self.feature_size = feature_size
            self.agg = ConvGRU1d(input_size=feature_size, hidden_size=feature_size, kernel_size=3,
                                 num_layers=1, device=device)
            self.predictor = nn.Sequential(
                nn.Conv1d(feature_size, feature_size, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(feature_size, feature_size, kernel_size=1, padding=0, bias=True)
            )
        elif network == 'r2d':
            self.encoder = R2DNet(input_channels, hidden_channels, feature_dim, stride=[(2, 2), (1, 1), (1, 1), (1, 1)],
                                  final_fc=False)
            feature_size = self.encoder.feature_size
            self.feature_size = feature_size
            self.agg = ConvGRU2d(input_size=feature_size, hidden_size=feature_size, kernel_size=3,
                                 num_layers=1, device=device)
            self.predictor = nn.Sequential(
                nn.Conv2d(feature_size, feature_size, kernel_size=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_size, feature_size, kernel_size=1, padding=0, bias=True)
            )
        else:
            raise ValueError
        # self.gru = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)

        self.relu = nn.ReLU(inplace=True)
        self.targets = None

        self._initialize_weights(self.agg)
        self._initialize_weights(self.predictor)

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        batch_size, num_epoch, channel, time_len = x.shape
        x = x.view(batch_size * num_epoch, channel, time_len)
        feature = self.encoder(x)
        if self.network == 'r1d':
            feature = F.avg_pool1d(feature, kernel_size=(1,), stride=(1,))
        else:
            feature = F.avg_pool2d(feature, kernel_size=(1, 1), stride=(1, 1))
        last_size = np.prod(feature.shape) // batch_size // num_epoch // self.feature_size
        assert batch_size * num_epoch * self.feature_size * last_size == np.prod(feature.shape)
        feature = feature.view(batch_size, num_epoch, self.feature_size, last_size)
        feature_relu = self.relu(feature)
        # feature_trans = feature.transpose(0, 2).contiguous()

        # # Get context feature
        # h_0 = self.gru.init_hidden(batch_size)
        # # out: (batch, num_seq, hidden_size)
        # # h_n: (num_layers, batch, hidden_size)
        # out, h_n = self.gru(feature[:, :-self.pred_steps, :], h_0)
        #
        # # Get predictions
        # pred = []
        # h_next = h_n
        # c_next = out[:, -1, :].squeeze(1)
        # for i in range(self.pred_steps):
        #     z_pred = self.predictor(c_next)
        #     pred.append(z_pred)
        #     c_next, h_next = self.gru(z_pred.unsqueeze(1), h_next)
        #     c_next = c_next[:, -1, :].squeeze(1)
        # pred = torch.stack(pred, 1)  # (batch, pred_step, feature_dim)
        # pred = pred.contiguous()

        ### aggregate, predict future ###
        _, hidden = self.agg(feature_relu[:, 0:num_epoch - self.pred_steps, :].contiguous())
        hidden = hidden[:, -1, :]  # after tanh, (-1,1). get the hidden state of last layer, last time step

        pred = []
        for i in range(self.pred_steps):
            # sequentially pred future
            p_tmp = self.predictor(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]
        pred = torch.stack(pred, 1)

        # Feature: (batch_size, num_epoch, feature_size, last_size)
        # Pred: (batch_size, pred_steps, feature_size, last_size)
        feature = feature.permute(0, 1, 3, 2).contiguous()
        feature = feature.view(batch_size * num_epoch * last_size, self.feature_size)
        feature = F.normalize(feature, p=2, dim=1)
        feature = feature.view(batch_size, num_epoch, last_size, self.feature_size)

        pred = pred.permute(0, 1, 3, 2).contiguous()
        pred = pred.view(batch_size * self.pred_steps * last_size, self.feature_size)
        pred = F.normalize(pred, p=2, dim=1)
        pred = pred.view(batch_size, self.pred_steps, last_size, self.feature_size)

        # Compute scores
        # logits = torch.einsum('ijk,kmn->ijmn', [pred, feature])  # (batch, pred_step, num_seq, batch)
        # logits = logits.view(batch_size * self.pred_steps, num_epoch * batch_size)

        logits = torch.einsum('ijkl,mnql->ijkqnm', [feature, pred])
        # print('3. Logits: ', logits.shape)
        logits = logits.view(batch_size * num_epoch * last_size, last_size * self.pred_steps * batch_size)
        logits /= self.temperature

        if self.targets is None:
            targets = torch.zeros(batch_size, num_epoch, last_size, last_size, self.pred_steps, batch_size)
            for i in range(batch_size):
                for j in range(last_size):
                    for k in range(self.pred_steps):
                        targets[i, num_epoch - self.pred_steps + k, j, j, k, i] = 1
            targets = targets.view(batch_size * num_epoch * last_size, last_size * self.pred_steps * batch_size)
            targets = targets.argmax(dim=1)
            targets = targets.cuda(device=self.device)
            self.targets = targets

        # if self.targets is None:
        #     targets = torch.zeros(batch_size, self.pred_steps, num_epoch, batch_size).long()
        #     for i in range(batch_size):
        #         for j in range(self.pred_steps):
        #             targets[i, j, num_epoch - self.pred_steps + j, i] = 1
        #     targets = targets.view(batch_size * self.pred_steps, num_epoch * batch_size)
        #     targets = targets.argmax(dim=1)
        #     self.targets = targets.cuda(self.device)

        return logits, self.targets

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class DPCClassifier(nn.Module):
    def __init__(self, network, input_channels, hidden_channels, feature_dim, pred_steps, num_class,
                 use_l2_norm, use_dropout, use_batch_norm, device):
        super(DPCClassifier, self).__init__()

        self.network = network
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.device = device
        self.use_l2_norm = use_l2_norm
        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm

        if network == 'r1d':
            self.encoder = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                                  final_fc=False)
            feature_size = self.encoder.feature_size
            self.feature_size = feature_size
            self.agg = ConvGRU1d(input_size=feature_size, hidden_size=feature_size, kernel_size=3,
                                 num_layers=1, device=device)
        elif network == 'r2d':
            self.encoder = R2DNet(input_channels, hidden_channels, feature_dim, stride=[(2, 2), (1, 1), (1, 1), (1, 1)],
                                  final_fc=False)
            feature_size = self.encoder.feature_size
            self.feature_size = feature_size
            self.agg = ConvGRU2d(input_size=feature_size, hidden_size=feature_size, kernel_size=3,
                                 num_layers=1, device=device)
        else:
            raise ValueError

        self.relu = nn.ReLU(inplace=True)

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(self.feature_size))
        if use_dropout:
            final_fc.append(nn.Dropout(0.5))
        final_fc.append(nn.Linear(feature_size, num_class))
        self.final_fc = nn.Sequential(*final_fc)

        # self._initialize_weights(self.final_fc)

    def forward(self, x):
        batch_size, num_epoch, channel, time_len = x.shape
        x = x.view(batch_size * num_epoch, channel, time_len)
        feature = self.encoder(x)
        feature = self.relu(feature)
        if self.network == 'r1d':
            feature = F.avg_pool1d(feature, kernel_size=(1,), stride=(1,))
        else:
            feature = F.avg_pool2d(feature, kernel_size=(1, 1), stride=(1, 1))
        last_size = np.prod(feature.shape) // batch_size // num_epoch // self.feature_size
        assert batch_size * num_epoch * self.feature_size * last_size == np.prod(feature.shape)
        feature = feature.view(batch_size, num_epoch, self.feature_size, last_size)

        context, _ = self.agg(feature)
        context = context[:, -1, :]
        # print('1. Context: ', context.shape)
        context = F.avg_pool1d(context, (last_size,), stride=1).squeeze()

        # print('2. Context: ', context.shape)

        if self.use_l2_norm:
            context = F.normalize(context, p=2, dim=1)

        out = self.final_fc(context)

        # print('3. Out: ', out.shape)

        return out

    # def _initialize_weights(self, module):
    # for m in module.modules():
    #     if isinstance(m, nn.BatchNorm1d):
    # for name, param in module.named_parameters():
    #         param.weight.data.fill_(1)
    #         param.bias.data.zero_()
    # else:
    #     for name, param in m.named_parameters():
    #         if 'bias' in name:
    #             nn.init.constant_(param, 0.0)
    #         elif 'weight' in name:
    #             nn.init.orthogonal_(param, 1)
