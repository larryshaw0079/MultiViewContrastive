"""
@Time    : 2020/11/10 21:30
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : coclr.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch
import torch.nn.functional as F

from .moco import Moco
from ..backbone.resnet import R1DNet, R2DNet


class CoCLR(Moco):
    '''
    CoCLR: using another view of the data to define positive and negative pair
    Still, use MoCo to enlarge the negative pool
    '''

    def __init__(self, network='r1d', in_channel=2, mid_channel=16, dim=128, K=2048, m=0.999, T=0.07, topk=5,
                 reverse=False):
        '''
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        '''
        super(CoCLR, self).__init__(network, in_channel, mid_channel, dim, K, m, T)

        self.topk = topk

        if network == 'r1d':
            self.sampler = R1DNet(in_channel, mid_channel, dim, stride=2)
        elif network == 'r2d':
            self.sampler = R2DNet(in_channel, mid_channel, dim, stride=(2, 2))
        else:
            raise ValueError

        for param_s in self.sampler.parameters():
            param_s.requires_grad = False  # not update by gradient

        # create another queue, for the second view of the data
        self.register_buffer("queue_second", torch.randn(dim, K))
        self.queue_second = F.normalize(self.queue_second, dim=0)

        # for handling sibling videos, e.g. for UCF101 dataset
        self.register_buffer("queue_vname", torch.ones(K, dtype=torch.long) * -1)
        # for monitoring purpose only
        self.register_buffer("queue_label", torch.ones(K, dtype=torch.long) * -1)

        self.queue_is_full = False
        self.reverse = reverse

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_second, vnames):
        # gather keys before updating queue
        #         keys = concat_all_gather(keys)
        #         keys_second = concat_all_gather(keys_second)
        #         vnames = concat_all_gather(vnames)
        # labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_second[:, ptr:ptr + batch_size] = keys_second.T
        self.queue_vname[ptr:ptr + batch_size] = vnames
        self.queue_label[ptr:ptr + batch_size] = torch.ones_like(vnames)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x1, x2, f1, f2, k_vsource):
        '''Output: logits, targets'''
        (B, *_) = x1.shape

        if self.reverse:
            x1, f1 = f1, x1
            x2, f2 = f2, x2

            # compute query features
        q = self.encoder_q(x1)  # queries: B,C
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

            # compute key feature for second view
            kf = self.sampler(f2)  # keys: B,C,1,1,1
            kf = F.normalize(kf, dim=1)
            kf = kf.view(B, self.dim)

        # if queue_second is full: compute mask & train CoCLR, else: train InfoNCE

        # compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: N,(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # mask: binary mask for positive keys
        # handle sibling videos, e.g. for UCF101. It has no effect on K400
        mask_source = k_vsource.unsqueeze(1) == self.queue_vname.unsqueeze(0)  # B,K
        mask = mask_source.clone()

        if not self.queue_is_full:
            self.queue_is_full = torch.all(self.queue_label != -1)

        if self.queue_is_full:
            print('\n===== queue is full now =====')

        if self.queue_is_full and (self.topk != 0):
            mask_sim = kf.matmul(self.queue_second.clone().detach())
            mask_sim[mask_source] = - np.inf  # mask out self (and sibling videos)
            _, topkidx = torch.topk(mask_sim, self.topk, dim=1)
            topk_onehot = torch.zeros_like(mask_sim)
            topk_onehot.scatter_(1, topkidx, 1)
            mask[topk_onehot.bool()] = True

        mask = torch.cat([torch.ones((mask.shape[0], 1), dtype=torch.long, device=mask.device).bool(),
                          mask], dim=1)

        # dequeue and enqueue
        if in_train_mode: self._dequeue_and_enqueue(k, kf, k_vsource)

        return logits, mask.detach()
