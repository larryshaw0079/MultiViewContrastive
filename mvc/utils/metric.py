"""
@Time    : 2020/11/9 16:54
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : metric.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def logits_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def get_performance(scores: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(scores, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')

    cm = confusion_matrix(labels, predictions)
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)

    return {'accuracy': accuracy, 'accuracy_per_class': accuracy_per_class.tolist(),
            'f1_micro': f1_micro, 'f1_macro': f1_macro}
