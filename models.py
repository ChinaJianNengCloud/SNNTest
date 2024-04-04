import torch
from torch import nn
from spikingjelly.activation_based import neuron, layer, surrogate


class SNN(nn.Module):
    def __init__(self, tau, features=1024):
        super().__init__()

        self.mlp = nn.Sequential(
            layer.Linear(features, features // 4, step_mode='m'),
            layer.Dropout(step_mode='m'),
            layer.Linear(features // 4, features, step_mode='m'),
            # layer.Dropout(step_mode='m'),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m'),
            # layer.Linear(features, features // 4, step_mode='m'),
            # layer.Dropout(step_mode='m'),
            # layer.Linear(features // 4, features, step_mode='m'),
            # layer.Dropout(step_mode='m'),
            # neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), step_mode='m'),
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)
