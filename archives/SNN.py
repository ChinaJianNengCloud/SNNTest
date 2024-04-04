import torch
from torch import nn, Tensor
from spikingjelly.activation_based import surrogate


class SNNMultiStepMode(nn.Module):
    def __init__(self, v_threshold=1, v_reset=None, surrogate_function=surrogate.Sigmoid(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.v = None
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function

    def reset(self):
        self.v = None

    def neuronal_charge(self, x):
        return self.v + x

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike) * self.v + spike * self.v_reset

    def forward(self, x: Tensor):
        # x.shape:(batch, sequence, feature)
        batch_size, seq_size, feature_size = x.size()
        # 根据输入的神经元生成初始化的v
        if self.v is None:
            self.v = torch.zeros((batch_size, feature_size))
            self.v.to(x.device)
        elif self.v.size() != torch.Size((batch_size, feature_size)):
            raise ValueError('Size diff! Please reset network.')
        hidden_seq = []
        for t in range(seq_size):
            x_t = x[:, t, :]
            self.neuronal_charge(x_t)
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            hidden_seq += [spike]

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq
