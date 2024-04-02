import click
import torch
import tqdm
import os
import time

from torch.cuda.amp import autocast
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler
from models import SNN
from dataset import SNNDataset
from utils import setup_seed, datetime_now_str, count_metrics_binary_classification, LossEarlyStopping

records_dir = './'
logs_dir = 'logs'
checkpoints_dir = 'checkpoints'


@click.command()
@click.argument('train_data_dir_path')
@click.argument('test_data_dir_path')
@click.option('--train_dir_prefix', default=datetime_now_str(), type=str, help='train dir prefix')
@click.option('--epochs', type=int, default=100)
@click.option('--lr', type=float, default=1e-4)
@click.option('--step_size', default=10, type=int, help='step size')
@click.option('--gamma', default=0.1, type=float, help='gamma')
@click.option('--use_early_stopping', default=True, help='if use early stopping')
@click.option('--early_stopping_step', default=7, type=int, help='early stopping step')
@click.option('--early_stopping_delta', default=0, type=int, help='early stopping delta')
@click.option('--batch_size', type=int, help='batch size', default=8)
@click.option('--drop_last', type=bool, default=True)
@click.option('--num_workers', type=int, default=4)
@click.option('--pin_memory', type=bool, default=True)
@click.option('--persistent_workers', type=bool, default=True)
@click.option('--log_interval', default=1, type=int, help='save metrics interval')
@click.option('--save_interval', default=10, type=int, help='save wts interval')
def main(train_data_dir_path, test_data_dir_path, train_dir_prefix,
         epochs, lr, step_size, gamma, batch_size,
         use_early_stopping: bool, early_stopping_step: int, early_stopping_delta: int,
         drop_last, num_workers, pin_memory, persistent_workers,
         log_interval, save_interval):
    setup_seed(2024)
    print(f'当前时间:{train_dir_prefix}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('current device: ', device)
    # 创建网络
    net = SNN(2.0)
    net.to(device)
    criterion = nn.BCELoss()
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # 早停策略
    loss_early_stopping = LossEarlyStopping(patience=early_stopping_step, delta=early_stopping_delta)
    scaler = GradScaler()
    log_dir_path = os.path.join(records_dir, train_dir_prefix, logs_dir)
    writer = SummaryWriter(log_dir=log_dir_path)
    checkpoint_path = os.path.join(records_dir, train_dir_prefix, checkpoints_dir)
    best_model_checkpoints_path = os.path.join(checkpoint_path, 'best_model_checkpoints.pth')
    train_data_loader = DataLoader(SNNDataset(train_data_dir_path), batch_size=batch_size,
                                   drop_last=drop_last, num_workers=num_workers, pin_memory=pin_memory,
                                   persistent_workers=persistent_workers, shuffle=True)
    test_data_loader = DataLoader(SNNDataset(test_data_dir_path), batch_size=batch_size,
                                  drop_last=drop_last, pin_memory=pin_memory,
                                  persistent_workers=persistent_workers)
    data_loaders = {'train': train_data_loader, 'valid': test_data_loader}
    best_mcc = 0
    start = time.time()
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                # 训练
                net.train()
            else:
                # 验证
                net.eval()
            running_loss = 0.0
            y_true, y_pred = [], []
            for data in tqdm.tqdm(data_loaders[phase]):
                # data.shape:(batch, time, feature)
                # 转为[T, N, *]
                data = data.transpose(0, 1)
                input_data = data[:30, :, :].to(device)
                target = data[30:, :, :, ].to(device)
                y_true += target.reshape(-1).tolist()
                # 梯度清零
                optimizer.zero_grad()
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        output_data = net(input_data)
                        loss = criterion(output_data, target)
                        # loss 带权重
                        weight = torch.zeros_like(target).float().to(device)
                        weight = torch.fill_(weight, 0.2)
                        weight[target > 0] = 0.8
                        loss = torch.mean(weight * loss)
                        y_pred += output_data.reshape(-1).tolist()
                    if phase == 'train':
                        if not scaler:
                            loss.backward()
                            optimizer.step()
                        else:
                            # Scales loss，为了梯度放大
                            scaler.scale(loss).backward()
                            # scaler.step() 首先把梯度的值unscale回来.
                            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                            scaler.step(optimizer)
                            # 准备着，看是否要增大scaler
                            scaler.update()
                running_loss += loss.item()
            # 计算损失
            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            # 计算指标
            all_metrics = count_metrics_binary_classification(y_true, y_pred)
            if phase == 'valid':
                if all_metrics['f1'] > best_mcc:
                    best_mcc = all_metrics['mcc']
                    # 保存最好的模型
                    torch.save({
                        'epoch': epoch,
                        'model': net.state_dict(),
                        'mcc': best_mcc,
                        'optimizer': optimizer.state_dict()
                    }, best_model_checkpoints_path)
                if use_early_stopping:
                    loss_early_stopping(epoch_loss)
            # 记录
            if epoch % log_interval == 0:
                writer.add_scalars('Loss', {
                    phase: epoch_loss,
                }, epoch)
                writer.add_scalars('acc', {
                    phase: all_metrics['acc'],
                }, epoch)
                writer.add_scalars('mcc', {
                    phase: all_metrics['mcc'],
                }, epoch)
                writer.add_scalars('precision', {
                    phase: all_metrics['precision'],
                }, epoch)
                writer.add_scalars('recall', {
                    phase: all_metrics['recall'],
                })
                writer.add_scalars('f1', {
                    phase: all_metrics['f1'],
                })
                writer.add_scalars('tpr', {
                    phase: all_metrics['tpr'],
                })
                writer.add_scalars('fpr', {
                    phase: all_metrics['fpr'],
                })
                writer.add_scalars('ks', {
                    phase: all_metrics['ks'],
                }, epoch)
                writer.add_scalars('sp', {
                    phase: all_metrics['sp'],
                }, epoch)
                writer.add_scalars('auc', {
                    phase: all_metrics['auc_score'],
                }, epoch)
                writer.flush()
            time_elapsed = time.time() - start
            print(
                f"Epoch {epoch + 1}/{epochs} | {phase} | Loss: {epoch_loss:.4f} | best MCC: {best_mcc}\n"
                f"acc: {all_metrics['acc']:.4f} | precision: {all_metrics['precision']:.4f} | recall: {all_metrics['recall']:.4f} | \n"
                f"f1: {all_metrics['f1']:.4f} | mcc: {all_metrics['mcc']:.4f} | sp: {all_metrics['sp']:.4f} | \n"
                f"tpr: {all_metrics['tpr']:.4f} | fpr: {all_metrics['fpr']:.4f} | ks: {all_metrics['ks']:.4f} | \n"
                f"Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
        scheduler.step()
        if epoch % step_size == 0:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': net.state_dict(),
                'mcc': all_metrics['mcc'],
                'optimizer': optimizer.state_dict()
            }, os.path.join(checkpoints_dir, f'epoch_{epoch}_model_checkpoints.pth'))
        if use_early_stopping and loss_early_stopping.early_stop:
            break
    # 训练全部完成
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best MCC: {:4f}'.format(best_mcc))
    # 加载最佳模型权重，最后做一次总的测试
    print('加载最佳模型权重，最后做一次总的测试')
    net.load_state_dict(torch.load(best_model_checkpoints_path)['model'])
    net.eval()
    y_true, y_pred = [], []
    for data in tqdm.tqdm(test_data_loader):
        # data.shape:(batch, time, feature)
        # 转为[T, N, *]
        data = data.transpose(0, 1)
        input_data = data[:30, :, :].to(device)
        target = data[30:, :, :, ].to(device)
        y_true += target.reshape(-1).tolist()
        with torch.no_grad():
            output_data = net(input_data)
            y_pred += output_data.reshape(-1).tolist()
        all_metrics = count_metrics_binary_classification(y_true, y_pred)
        print(
            f"acc: {all_metrics['acc']:.4f} | precision: {all_metrics['precision']:.4f} | recall: {all_metrics['recall']:.4f} | \n"
            f"f1: {all_metrics['f1']:.4f} | mcc: {all_metrics['mcc']:.4f}  | sp: {all_metrics['sp']:.4f} | \n"
            f"tpr: {all_metrics['tpr']:.4f} | fpr: {all_metrics['fpr']:.4f} | ks: {all_metrics['ks']:.4f} | \n"
        )


if __name__ == '__main__':
    main()
