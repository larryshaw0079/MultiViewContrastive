"""
@Time    : 2020/11/9 16:54
@Author  : Xiao Qinfeng
@Email   : qfxiao@bjtu.edu.cn
@File    : misc.py
@Software: PyCharm
@Desc    : 
"""
from sklearn.metrics import accuracy_score, f1_score


def get_performance(predictions, labels):
    accuracy = accuracy_score(labels, predictions)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')

    return {'accuracy': accuracy, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
