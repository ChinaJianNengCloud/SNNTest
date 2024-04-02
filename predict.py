import os.path

import click
import torch
from models import SNN


@click.command()
@click.argument('checkpoints_path')
@click.argument('file_path')
@click.option('--row', type=int, default=1024)
@click.option('--col', type=int, default=60)
@click.option('--output_dir_path', type=str, default='./')
def main(checkpoints_path, file_path, row, col, output_dir_path):
    file_name = os.path.basename(file_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SNN(2.0)
    net.load_state_dict(torch.load(checkpoints_path)['model'])
    with open(file_path) as f:
        lines = f.read().split('\n')
    data = torch.zeros((row, col))
    for line in lines:
        row, col, _ = line.split(' ')
        data[int(row), int(col)] = 1
    # 转成(t, features)
    data = data.t()
    # 增加batch维度
    data = data.unsqueeze(0)
    # 转为[T, N, *]
    data = data.transpose(0, 1)
    input_data = data[:30, :, :].to(device)
    target = data[30:, :, :, ].to(device)
    net.eval()
    with torch.no_grad():
        output_data = net(input_data)
    # todo:画图和保存


if __name__ == '__main__':
    main()
