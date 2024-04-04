import click
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataset import SNNDataset
from models import SNN
from utils.data_handler import batch_data_reformat


@click.command()
@click.argument('checkpoints_path')
@click.argument('test_data_dir_path')
@click.option('--row', type=int, default=1024)
@click.option('--col', type=int, default=60)
@click.option('--group_size', type=int, default=10)
@click.option('--batch_size', type=int, help='batch size', default=1000)
@click.option('--drop_last', type=bool, default=True)
@click.option('--num_workers', type=int, default=4)
@click.option('--pin_memory', type=bool, default=True)
@click.option('--persistent_workers', type=bool, default=True)
@click.option('--output_dir_path', type=str, default='./')
def main(checkpoints_path, test_data_dir_path, row, col,
         group_size, batch_size,
         drop_last, num_workers, pin_memory, persistent_workers,
         output_dir_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SNN(2.0)
    net.to(device)
    net.load_state_dict(torch.load(checkpoints_path)['model'])
    test_data_loader = DataLoader(SNNDataset(test_data_dir_path, row, col), batch_size=batch_size,
                                  drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory,
                                  persistent_workers=persistent_workers)
    net.eval()
    for data in test_data_loader:
        data_reformat = batch_data_reformat(data, group_size)
        input_data = data_reformat[:data_reformat.shape[0] // 2, :, :].to(device)
        target = data_reformat[data_reformat.shape[0] // 2:, :, :, ].to(device)
        with torch.no_grad():
            output_data = net(input_data)
        break
    target_np = target[:, 0, :].cpu().numpy().T
    output_data_np = output_data[:, 0, :].cpu().numpy().T
    # Create a figure with two subplots, side by side.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))  # 1 row, 2 columns

    # Plot for original trial_data
    ax1.imshow(target_np, aspect='auto', origin='lower', interpolation='none', cmap='binary')
    ax1.set_xlabel('Time Point')
    ax1.set_ylabel('Channel')

    # Plot for predicted trial_data
    ax2.imshow(output_data_np, aspect='auto', origin='lower', interpolation='none', cmap='binary')
    ax2.set_xlabel('Time Point')
    ax2.set_ylabel('Channel')

    # Show the combined plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
