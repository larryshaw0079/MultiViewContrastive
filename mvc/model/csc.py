"""
@Time    : 2021/1/2 23:14
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : csc.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import R1DNet, ResNet, GRU


class CSC(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, pred_steps, reverse, use_temperature, temperature,
                 K, prop_iter, num_prop, device):
        super(CSC, self).__init__()

        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.reverse = reverse
        self.use_temperature = use_temperature
        self.temperature = temperature
        self.K = K
        self.prop_iter = prop_iter
        self.num_prop = num_prop
        self.device = device

        if reverse:
            self.encoder = ResNet(input_channels=input_channels, num_classes=feature_dim)
            self.sampler = R1DNet()
            self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
            self.predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )
        else:
            self.encoder = R1DNet()
            self.sampler = ResNet(input_channels=input_channels, num_classes=feature_dim)
            self.agg = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2, device=device)
            self.predictor = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, feature_dim)
            )

        self.relu = nn.ReLU(inplace=True)
        self.targets_pred = None

        self.register_buffer("queue_first", torch.randn(feature_dim, K))
        self.queue_first = F.normalize(self.queue_first, dim=0)

        self.register_buffer("queue_second", torch.randn(feature_dim, K))
        self.queue_second = F.normalize(self.queue_second, dim=0)

        self.register_buffer("queue_idx", torch.ones(K, dtype=torch.long) * -1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_is_full = False

    def _dequeue_and_enqueue(self, feature_q, feature_k, idx):
        pass

    def forward(self, x1, x2, idx):
        if self.reverse:
            x1, x2 = x2, x1

        assert x1.shape[:2] == x2.shape[:2] and x1.shape[:2] == idx.shape[:2]
        (B1, num_epoch, *epoch_shape1) = x1.shape
        (B2, num_epoch, *epoch_shape2) = x2.shape

        x1 = x1.view(B1 * num_epoch, *epoch_shape1)
        feature_q = self.encoder(x1)
        feature_q = feature_q.view(B1, num_epoch, self.feature_dim)

        x2 = x2.view(B2 * num_epoch, *epoch_shape2)
        feature_k = self.sampler(x2)
        feature_k = feature_k.view(B2, num_epoch, self.feature_dim)

        feature_relu = self.relu(feature_q)

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
            feature_k = F.normalize(feature_k, p=2, dim=-1)
            pred = F.normalize(pred, p=2, dim=-1)

        logits_pred = torch.einsum('ijk,mnk->ijnm', [feature_q, pred])
        logits_pred = logits_pred.view(B1 * num_epoch, self.pred_steps * B1)

        logits_mem = torch.einsum('ijk,km->ijm', [feature_q, self.queue_first.close().detach()])
        logits_mem = logits_mem.view(B1 * num_epoch, -1)

        # logits = torch.cat([logits_pred, logits_mem], dim=-1) # (B*num_epoch, pred_steps*B+K)

        if self.use_temperature:
            logits_pred /= self.temperature
            logits_mem /= self.temperature

        if self.targets_pred is None:
            targets_pred = torch.zeros(B1, num_epoch, self.pred_steps, B1)
            for i in range(B1):
                for j in range(self.pred_steps):
                    targets_pred[i, num_epoch - self.pred_steps + j, j, i] = 1
            targets_pred = targets_pred.view(B1 * num_epoch, self.pred_steps * B1)
            targets_pred = targets_pred.argmax(dim=1)
            targets_pred = targets_pred.cuda(device=self.device)
            self.targets_pred = targets_pred

        targets_mem = torch.zeros(B1, num_epoch, self.K)
        for i in range(self.prop_iter):
            mem_sim = torch.einsum('ijk,km->ijm', [feature_k, self.queue_second.clone().detach()])
            mem_sim = mem_sim.view(B1 * num_epoch, self.K)
            _, topk_idx = torch.topk(mem_sim, self.num_prop, dim=1)

        targets_mem = targets_mem.cuda(self.device)

        self._dequeue_and_enqueue(feature_q, feature_k, idx)

        return logits_pred, logits_mem, targets_pred, targets_mem
