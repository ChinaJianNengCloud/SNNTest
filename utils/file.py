import torch


def read_file(file_path, row, col):
    with open(file_path) as f:
        # 注意过滤空行
        lines = list(filter(None, f.read().split('\n')))
    data = torch.zeros((row, col))
    for line in lines:
        row, col, _ = line.split('\t')
        data[int(row) - 1, int(col) - 1] = 1
    # 转成(t, features)
    return data.t()
