import torch
from spikingjelly.activation_based import neuron
print(torch.Tensor([[1,2],[3,4]])-1)

if_layer = neuron.IFNode()
x = torch.rand(size=[2, 3])
print(if_layer(x))
print(f'x.shape={x.shape}, if_layer.v.shape={if_layer.v.shape}')

print(if_layer.v.size()==torch.Size((2,3)))
# # x.shape=torch.Size([2, 3]), if_layer.v.shape=torch.Size([2, 3])
# if_layer.reset()
#
# x = torch.rand(size=[4, 5, 6])
# if_layer(x)
#
# print(f'x.shape={x.shape}, if_layer.v.shape={if_layer.v.shape}')
# # x.shape=torch.Size([4, 5, 6]), if_layer.v.shape=torch.Size([4, 5, 6])
# if_layer.reset()
# x = torch.as_tensor([0.02])
# T = 150
# s_list = []
# v_list = []
# for t in range(T):
#     s_list.append(if_layer(x))
#     v_list.append(if_layer.v)
# print(s_list)
# print(v_list)
