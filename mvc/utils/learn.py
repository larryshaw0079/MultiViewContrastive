"""
@Time    : 2020/12/16 16:51
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : learn.py
@Software: PyCharm
@Desc    : 
"""
import math


def adjust_learning_rate(optimizer, lr, epoch, total_epochs, args):
    """Decay the learning rate based on schedule"""
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
    else:  # stepwise lr schedule
        for milestone in args.lr_schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
