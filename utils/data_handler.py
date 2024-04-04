import torch


def batch_data_reformat(batch_data: torch.Tensor, group_size: int = 10):
    # 原始数据 shape:(batch, time, feature)
    # 转为[T, N, *]
    data = batch_data.transpose(0, 1)
    if group_size > 1:
        # 将 N 的维度每 N 组进行数据合并
        data_reshaped = data.view(data.shape[0], -1, group_size, data.shape[2])
        # 计算每个分组中是否有1
        group_sum = data_reshaped.sum(dim=2)
        # 将结果中大于0的元素置为1
        data_result = torch.where(group_sum > 0, torch.tensor(1.0), torch.tensor(0.0))
    else:
        data_result = data
    return data_result
