import torch
from sklearn.metrics import confusion_matrix


def cal_metrics_binary_classification(y_true: list, y_pred: list):
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
    mcc = ((tp * tn) - (fp * fn)) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return {'tpr': tpr, 'fpr': fpr, 'ks': ks, 'sp': sp, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def cal_metrics_binary_classification_limit(y_true: list, y_pred: list):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp + tn) / (tp + fn + tn + fp)
    recall = tp / (tp + fn)
    return {'acc': acc, 'recall': recall}


def cal_model_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    # 注意这里要reshape(-1)变成一维
    model_metrics = cal_metrics_binary_classification_limit(
        y_true.reshape(-1).tolist(),
        y_pred.reshape(-1).tolist()
    )
    # 等价于下面的操作
    # model_metrics_all = []
    # for i in range(y_true.shape[1]):
    #     y_true_tmp = y_true[:, i, :]
    #     y_pred_tmp = y_pred[:, i, :]
    #     # attention:一个文件一个1没有是不可能的，不需要判断
    #     # if torch.all(y_true_tmp==0):
    #     #     continue
    #     # else:
    #     model_metrics_all.append(
    #         cal_metrics_binary_classification_limit(
    #             y_true_tmp.reshape(-1).tolist(),
    #             y_pred_tmp.reshape(-1).tolist())
    #     )
    # model_metrics_keys = model_metrics_all[0].keys()
    # model_metrics_length = len(model_metrics_all)
    # model_metrics = {key: sum([item[key] for item in model_metrics_all]) / model_metrics_length for key in
    #                  model_metrics_keys}
    return model_metrics


def cal_channel_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    channel_metrics_all = {}
    for i in range(y_true.shape[2]):
        # 循环计算速度太慢，取巧，理论上来说等效
        # batch_metrics_all = []
        # for j in range(y_true.shape[1]):
        #     y_true_tmp = y_true[:, j, i]
        #     y_pred_tmp = y_pred[:, j, i]
        #     if torch.all(y_true_tmp == 0):
        #         continue
        #     else:
        #         batch_metrics_all.append(
        #             cal_metrics_binary_classification_limit(y_true_tmp.tolist(), y_pred_tmp.tolist()))
        # batch_metrics_keys = batch_metrics_all[0].keys()
        # batch_metrics_length = len(batch_metrics_all)
        # batch_metrics = {key: sum([item[key] for item in batch_metrics_all]) / batch_metrics_length for key in
        #                  batch_metrics_keys}
        y_true_tmp = y_true[:, :, i]
        y_pred_tmp = y_pred[:, :, i]
        if torch.all(y_true_tmp == 0):
            continue
        else:
            batch_metrics = cal_metrics_binary_classification_limit(
                y_true_tmp.reshape(-1).tolist(),
                y_pred_tmp.reshape(-1).tolist(),
            )
        channel_metrics_all[str(i)] = batch_metrics
    return channel_metrics_all


def cal_metrics(y_true: list, y_pred: list):
    # shape:([T ,N ,C])
    y_true = torch.cat(y_true, dim=1)
    y_pred = torch.cat(y_pred, dim=1)
    print('cal model metrics')
    model_metrics = cal_model_metrics(y_true, y_pred)
    print('cal channel metrics')
    channel_metrics = cal_channel_metrics(y_true, y_pred)
    return model_metrics, channel_metrics
