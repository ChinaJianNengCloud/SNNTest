from typing import Tuple

import torch
import math
from torch import nn, Tensor, init


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.U_i = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_u_i = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.b_v_i = nn.Parameter(torch.Tensor(self.hidden_size, 1))

        self.U_f = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_u_f = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.b_v_f = nn.Parameter(torch.Tensor(self.hidden_size, 1))

        self.U_o = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_u_o = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.b_v_o = nn.Parameter(torch.Tensor(self.hidden_size, 1))

        self.U_c = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b_u_c = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.b_v_c = nn.Parameter(torch.Tensor(self.hidden_size, 1))

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, x, states=None):
        # x.shape:(batch, sequence, feature)
        batch_size, seq_size, _ = x.size()

        # 上一次或者初始状态
        if states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device),
                        torch.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = states

        hidden_seq = []
        for t in range(seq_size):
            x_t = x[:, t, :]
            # b应该可以合并成一个
            i_t = torch.sigmoid(self.U_i @ x_t + self.b_u_i +
                                self.V_i @ h_t + self.b_v_i)
            f_t = torch.sigmoid(self.U_f @ x_t + self.b_u_f +
                                self.V_f @ h_t + self.b_v_f)
            o_t = torch.sigmoid(self.U_o @ x_t + self.b_u_o +
                                self.V_o @ h_t + self.b_v_o)
            g_t = torch.tanh(self.U_c @ x_t + self.b_u_c +
                             self.V_c @ h_t + self.b_v_c)
            # 更新
            c_t = (f_t * c_t +
                   i_t * g_t)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq += [h_t.unsqueeze(0)]
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
