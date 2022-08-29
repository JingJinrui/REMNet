import torch as t
from torch import nn
import torch.nn.functional as F


class convlstm_unit_remnet_fb(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, rec_length, bias=True):
        super(convlstm_unit_remnet_fb, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.rec_length = rec_length
        self.mem_gui_channels = int(240 / self.rec_length)
        self.padding = kernel_size // 2
        self.bias = bias
        self.conv = nn.Conv2d(self.in_channels + self.hidden_channels, self.hidden_channels * 5, self.kernel_size,
                              padding=self.padding, bias=self.bias)
        self.conv_c = nn.Conv2d(self.hidden_channels, self.mem_gui_channels, (1, 1), (1, 1), bias=self.bias)  # convert the cell state c-1 to the query vector
        self.conv_ssm = nn.ConvTranspose2d(self.mem_gui_channels, self.mem_gui_channels, (4, 4), (2, 2), (1, 1),
                                           bias=self.bias)  # match spatial size 32 to 64
        self.conv_fuse = nn.Conv2d(self.mem_gui_channels + self.hidden_channels, self.hidden_channels, (1, 1), (1, 1),
                                   bias=self.bias)  # 1x1 convolution kernel

    # c [batch_size, hidden_channels, height, width]
    # recalled_memory_features [batch_size, 240, 32, 32]
    # pertimestep_memory_guis [batch_size, mem_gui_channels, 32, 32]
    def pertimestep_perception_attention(self, c, recalled_memory_features):
        batch_size = c.size()[0]
        recalled_memory_features = recalled_memory_features.reshape(batch_size, self.rec_length, self.mem_gui_channels,
                                                                    32, 32)  # [batch_size, rec_length, mem_gui_channels, 32, 32]
        recalled_memory_features_averaged = F.adaptive_avg_pool3d(recalled_memory_features, (
        self.mem_gui_channels, 1, 1)).squeeze(4).squeeze(3)  # [batch_size, rec_length, mem_gui_channels]
        pertimestep_memory_guis = []
        query_vector = F.leaky_relu(self.conv_c(c), negative_slope=0.2, inplace=True)  # [batch_size, mem_gui_channels, height, width]
        query_vector_averaged = F.adaptive_avg_pool2d(query_vector, (1, 1)).squeeze(3).squeeze(2)  # [batch_size, mem_gui_channels]
        for i in range(batch_size):
            attention_weight = t.cosine_similarity(query_vector_averaged[i].repeat(self.rec_length, 1),
                                                   recalled_memory_features_averaged[i], dim=1)  # [rec_length]
            attention_weight = t.unsqueeze(t.unsqueeze(t.unsqueeze(F.softmax(attention_weight, dim=0), dim=1), dim=2),
                                           dim=3)
            pertimestep_memory_gui = t.sum(t.mul(recalled_memory_features[i], attention_weight), dim=0)  # [mem_gui_channels, 32, 32]
            pertimestep_memory_guis.append(pertimestep_memory_gui)
        pertimestep_memory_guis = t.stack(pertimestep_memory_guis, dim=0)
        return pertimestep_memory_guis

    # input:
    # x [batch_size, in_channels, height, width]
    # state (h, c)
    # h c [batch_size, hidden_channels, height, width]
    # recalled_memory_feature [batch_size, 240, 32, 32]
    # output:
    # state (h, c)
    # h c [batch_size, hidden_channels, height, width]
    def forward(self, x, state, recalled_memory_feature):
        h, c = state
        mt = self.pertimestep_perception_attention(c, recalled_memory_feature)
        mt = F.leaky_relu(self.conv_ssm(mt), negative_slope=0.2, inplace=True)
        i, f, g, o, lam = t.split(self.conv(t.cat((x, h), dim=1)), self.hidden_channels, dim=1)
        i = F.sigmoid(i)
        f = F.sigmoid(f)
        o = F.sigmoid(o)
        g = F.tanh(g)
        lam = F.sigmoid(lam)
        c = f * c + i * g
        h_new = o * F.tanh(self.conv_fuse(t.cat((c, mt), dim=1)))
        h = lam * h_new + (1.0 - lam) * h
        state = (h, c)
        return state
