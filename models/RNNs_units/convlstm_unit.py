import torch as t
from torch import nn


class convlstm_unit(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias=True):
        super(convlstm_unit, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(self.in_channels + self.hidden_channels, self.hidden_channels * 4, self.kernel_size, padding=self.padding, bias=self.bias)

    # input:
    # x [batch_size, in_channels, height, width]
    # state (h, c)
    # h c [batch_size, hidden_channels, height, width]
    # output:
    # state (h, c)
    # h c [batch_size, hidden_channels, height, width]
    def forward(self, x, state):
        h, c = state
        i, f, g, o = t.split(self.conv(t.cat((x, h), dim=1)), self.hidden_channels, dim=1)
        i = t.sigmoid(i)
        f = t.sigmoid(f)
        o = t.sigmoid(o)
        g = t.tanh(g)
        c = f * c + i * g
        h = o * t.tanh(c)
        state = (h, c)
        return state
