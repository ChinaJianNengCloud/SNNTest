import os

from torch.utils.data import Dataset

from utils.file import read_file


class SNNDataset(Dataset):
    def __init__(self, data_dir_path, row=1024, col=60):
        self.data_dir_path = data_dir_path
        self.row = row
        self.col = col

        self.file_paths = [os.path.join(data_dir_path, file_name)
                           for file_name in os.listdir(data_dir_path)]

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        return read_file(file_path, self.row, self.col)

    def __len__(self):
        return len(self.file_paths)
