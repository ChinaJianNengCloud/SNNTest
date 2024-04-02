import os

import torch
from torch.utils.data import Dataset


class SNNDataset(Dataset):
    def __init__(self, data_dir_path, row=1024, col=60):
        self.data_dir_path = data_dir_path
        self.row = row
        self.col = col

        self.file_paths = [os.path.join(data_dir_path, file_name)
                           for file_name in os.listdir(data_dir_path)]

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        with open(file_path) as f:
            lines = f.read().split('\n')
        data = torch.zeros((self.row, self.col))
        for line in lines:
            row, col, _ = line.split(' ')
            data[int(row), int(col)] = 1
        # 转成(t, features)
        return data.t()

    def __len__(self):
        return len(self.file_paths)
