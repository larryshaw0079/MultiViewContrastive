"""
@Time    : 2020/11/10 21:27
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : moco.py
@Software: PyCharm
@Desc    : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.resnet import R1DNet, R2DNet


class Moco(nn.Module):
    def __init__(self, network='r1d', in_channel=2, mid_channel=16, dim=128, K=2048, m=0.999, T=0.07):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(Moco, self).__init__()

        assert network in ['r1d', 'r2d']

        self.network = network
        self.dim = dim
        self.K = K
        self.m = m
        self.T = T

        # create the encoders (including non-linear projection head: 2 FC layers)
        if network == 'r1d':
            self.encoder_q = R1DNet(in_channel, mid_channel, dim, stride=2)
            self.encoder_k = R1DNet(in_channel, mid_channel, dim, stride=2)
        elif network == 'r2d':
            self.encoder_q = R2DNet(in_channel, mid_channel, dim, stride=(2, 2))
            self.encoder_k = R2DNet(in_channel, mid_channel, dim, stride=(2, 2))
        else:
            raise ValueError

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Notes: for handling sibling videos, e.g. for UCF101 dataset

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        '''Momentum update of the key encoder'''
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x1, x2):
        '''Output: logits, targets'''
        B, *_ = x1.shape

        # compute query features
        q = self.encoder_q(x1)
        q = F.normalize(q, dim=1)
        q = q.view(B, self.dim)

        in_train_mode = q.requires_grad

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if in_train_mode: self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            #             x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

            k = self.encoder_k(x2)  # keys: B,C,1,1,1
            k = F.normalize(k, dim=1)

            # undo shuffle
        #             k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        k = k.view(B, self.dim)

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (B, 1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # (B, K)

        # logits: B,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)  # (B, K+1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        if torch.cuda.is_available():
            labels = labels.cuda()

        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k)

        return logits, labels
