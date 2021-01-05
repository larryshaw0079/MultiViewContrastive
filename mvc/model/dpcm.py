"""
@Time    : 2020/12/29 16:48
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : dpcm.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import R1DNet, R2DNet, GRU, ResNet


class DPCMem(nn.Module):
    def __init__(self, network, input_channels, hidden_channels, feature_dim, pred_steps, use_temperature, temperature,
                 use_memory_pool=False, m=None, K=None, device='cuda'):
        super(DPCMem, self).__init__()

        self.network = network
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.use_memory_pool = use_memory_pool
        self.m = m
        self.K = K
        self.device = device

        if use_memory_pool:
            assert m is not None
            assert K is not None

        if network == 'r1d':
            self.encoder = R1DNet(input_channels, hidden_channels, feature_dim, stride=2, kernel_size=[7, 11, 11, 7],
                                  final_fc=True)
            if use_memory_pool:
                self.mirror_encoder = R1DNet(input_channels, hidden_channels, feature_dim, stride=2,
                                             kernel_size=[7, 11, 11, 7],
                                             final_fc=True)
            feature_size = self.encoder.feature_size
            self.feature_size = feature_size
            self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
            self.predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )
        elif network == 'r2d':
            self.encoder = R2DNet(input_channels, hidden_channels, feature_dim, stride=[(2, 2), (1, 1), (1, 1), (1, 1)],
                                  final_fc=True)
            if use_memory_pool:
                self.mirror_encoder = R2DNet(input_channels, hidden_channels, feature_dim,
                                             stride=[(2, 2), (1, 1), (1, 1), (1, 1)],
                                             final_fc=True)
            feature_size = self.encoder.feature_size
            self.feature_size = feature_size
            self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
            self.predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )
        elif network == 'r2d_img':
            self.encoder = ResNet(input_channels=input_channels, num_classes=feature_dim)
            if use_memory_pool:
                self.mirror_encoder = ResNet(input_channels=input_channels, num_classes=feature_dim)
            self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
            self.predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )
        else:
            raise ValueError
        # self.gru = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)

        if use_memory_pool:
            for param_q, param_k in zip(self.encoder.parameters(), self.mirror_encoder.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        self.relu = nn.ReLU(inplace=True)
        self.targets = None

        if use_memory_pool:
            self.register_buffer("queue", torch.randn(feature_dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self._initialize_weights(self.agg)
        self._initialize_weights(self.predictor)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder.parameters(), self.mirror_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = keys.view(-1, self.feature_dim)
        batch_size, *_ = keys.shape

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f'{self.K}, {batch_size}'  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        if self.network == 'r1d':
            batch_size, num_epoch, channel, time_len = x.shape
            x = x.view(batch_size * num_epoch, channel, time_len)
        else:
            batch_size, num_epoch, channel, freq_len, time_len = x.shape
            x = x.view(batch_size * num_epoch, channel, freq_len, time_len)
        feature_q = self.encoder(x)  # (batch_size, num_epoch, feature_size)
        feature_q = feature_q.view(batch_size, num_epoch, self.feature_dim)
        feature_relu = self.relu(feature_q)

        if self.use_memory_pool:
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

            feature_k = self.mirror_encoder(x)
            feature_k = feature_k.view(batch_size, num_epoch, self.feature_dim)

        out, h_n = self.agg(feature_relu[:, :-self.pred_steps, :].contiguous())

        # Get predictions
        pred = []
        h_next = h_n
        c_next = out[:, -1, :].squeeze(1)
        for i in range(self.pred_steps):
            z_pred = self.predictor(c_next)
            pred.append(z_pred)
            c_next, h_next = self.agg(z_pred.unsqueeze(1), h_next)
            c_next = c_next[:, -1, :].squeeze(1)
        pred = torch.stack(pred, 1)  # (batch, pred_step, feature_dim)
        # Compute scores
        pred = pred.contiguous()

        if self.use_temperature:
            feature_q = F.normalize(feature_q, p=2, dim=-1)
            pred = F.normalize(pred, p=2, dim=-1)
            if self.use_memory_pool:
                feature_k = F.normalize(feature_k, p=2, dim=-1)

        # feature (batch_size, num_epoch, feature_size)
        # pred (batch_size, pred_steps, feature_size)
        if self.use_memory_pool:
            # cat_feature (batch_size, num_epoch, last_size+memsize, feature_size)
            assert self.K % batch_size == 0  # For simplicity
            # cat_feature = torch.cat([feature_q, self.queue.T.view(batch_size, -1, self.feature_dim)], dim=-2)
            # logits = torch.einsum('ijk,mnk->ijnm', [cat_feature, pred])
            # logits = logits.view(batch_size * (num_epoch + self.memory_pool_size // batch_size),
            #                      self.pred_steps * batch_size)
            logits_pred = torch.einsum('ijk,mnk->ijnm', [feature_k, pred])
            logits_pred = logits_pred.view(batch_size * num_epoch, self.pred_steps * batch_size)

            logits_mem = torch.einsum('ki,mnk->inm', [self.queue.clone().detach(), pred])
            logits_mem = logits_mem.view(self.K, self.pred_steps * batch_size)

            logits = torch.cat([logits_pred, logits_mem], dim=0)
            logits = logits.t()
        else:
            logits = torch.einsum('ijk,mnk->ijnm', [feature_q, pred])
            logits = logits.view(batch_size * num_epoch, self.pred_steps * batch_size)

        if self.use_temperature:
            logits /= self.temperature

        if self.targets is None:
            if self.use_memory_pool:
                targets_pred = torch.zeros(batch_size, num_epoch, self.pred_steps, batch_size)
                for i in range(batch_size):
                    for j in range(self.pred_steps):
                        targets_pred[i, num_epoch - self.pred_steps + j, j, i] = 1
                targets_pred = targets_pred.view(batch_size * num_epoch, self.pred_steps * batch_size)
                targets_mem = torch.zeros(self.K, self.pred_steps * batch_size)
                targets = torch.cat([targets_pred, targets_mem], dim=0)
                targets = targets.t()
                targets = targets.argmax(dim=1)
                targets = targets.cuda(device=self.device)
                self.targets = targets
            else:
                targets = torch.zeros(batch_size, num_epoch, self.pred_steps, batch_size)
                for i in range(batch_size):
                    for j in range(self.pred_steps):
                        targets[i, num_epoch - self.pred_steps + j, j, i] = 1
                targets = targets.view(batch_size * num_epoch, self.pred_steps * batch_size)
                targets = targets.argmax(dim=1)
                targets = targets.cuda(device=self.device)
                self.targets = targets

        if self.use_memory_pool:
            self._dequeue_and_enqueue(feature_k)

        return logits, self.targets

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


class DPCMemClassifier(nn.Module):
    def __init__(self, network, input_channels, hidden_channels, feature_dim, pred_steps, num_class,
                 use_l2_norm, use_dropout, use_batch_norm, device):
        super(DPCMemClassifier, self).__init__()

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
                                  final_fc=True)
            feature_size = self.encoder.feature_size
            self.feature_size = feature_size
            self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
        elif network == 'r2d':
            self.encoder = R2DNet(input_channels, hidden_channels, feature_dim, stride=[(2, 2), (1, 1), (1, 1), (1, 1)],
                                  final_fc=True)
            feature_size = self.encoder.feature_size
            self.feature_size = feature_size
            self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
        elif network == 'r2d_img':
            self.encoder = ResNet(input_channels=input_channels, num_classes=feature_dim)
            self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
        else:
            raise ValueError

        self.relu = nn.ReLU(inplace=True)

        final_fc = []

        if use_batch_norm:
            final_fc.append(nn.BatchNorm1d(feature_dim))
        if use_dropout:
            final_fc.append(nn.Dropout(0.5))
        final_fc.append(nn.Linear(feature_dim, num_class))
        self.final_fc = nn.Sequential(*final_fc)

    def forward(self, x):
        if self.network == 'r1d':
            batch_size, num_epoch, channel, time_len = x.shape
            x = x.view(batch_size * num_epoch, channel, time_len)
        else:
            batch_size, num_epoch, channel, freq_len, time_len = x.shape
            x = x.view(batch_size * num_epoch, channel, freq_len, time_len)
        feature = self.encoder(x)
        feature = self.relu(feature)
        feature = feature.view(batch_size, num_epoch, self.feature_dim)

        context, _ = self.agg(feature)
        context = context[:, -1, :]
        # print('1. Context: ', context.shape)
        # print('2. Context: ', context.shape)

        if self.use_l2_norm:
            context = F.normalize(context, p=2, dim=1)

        out = self.final_fc(context)

        # print('3. Out: ', out.shape)

        return out
