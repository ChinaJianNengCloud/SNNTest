from datetime import datetime
import random
import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def setup_seed(seed: int):
    """
    保证每次实验的结果是一样的
    :param seed: 随机数种子
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def datetime_now_str() -> str:
    # 获取处理后的时间
    return datetime.now().strftime('%Y%m%d%H%M%S')


def count_metrics_binary_classification(y_true: list, y_pred: list):
    """
    计算指标（二分类）
    :param y_true: 真实值
    :param y_pred: 预测值
    :return:
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    ks = abs(tpr - fpr)
    sp = 1 - fpr
    acc = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return locals()


class LossEarlyStopping:
    """
    这个早停类不做除了是否早停外的其他操作（保存模型和权重等等）
    """

    def __init__(self, patience: int = 7, delta: float = 0, silent: bool = False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        patience：自上次模型在验证集上损失降低之后等待的时间，此处设置为7
        counter：计数器，当其值超过patience时候，使用early stopping
        best_score：记录模型评估的最好分数
        early_step：决定模型要不要early stop，为True则停
        val_loss_min：模型评估损失函数的最小值，默认为正无穷(np.Inf)
        delta：表示模型损失函数改进的最小值，当超过这个值时候表示模型有所改进
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        self.silent = silent

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if not self.silent:
                print(f'{self.__class__.__name__} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
