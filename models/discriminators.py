import torch as t
from torch import nn
import torch.nn.functional as F


class remnet_sequence_discriminator(nn.Module):
    def __init__(self):
        super(remnet_sequence_discriminator, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 32, (3, 5, 5), (2, 2, 2), (1, 2, 2), bias=True)
        self.conv3d_2 = nn.Conv3d(32, 64, (3, 3, 3), (2, 2, 2), (1, 1, 1), bias=True)
        self.conv3d_3 = nn.Conv3d(64, 128, (3, 3, 3), (2, 2, 2), (1, 1, 1), bias=True)
        self.conv3d_4 = nn.Conv3d(128, 256, (3, 3, 3), (2, 2, 2), (1, 1, 1), bias=True)
        self.linear = nn.Linear(256, 1)

    # input [out_seq_len, batch_size, 1, 256, 256]
    # output [batch_size]
    def forward(self, x):
        x = x.permute(1, 2, 0, 3, 4)
        x = F.avg_pool3d(x, (1, 2, 2), (1, 2, 2))
        x = F.leaky_relu(self.conv3d_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_3(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_4(x), negative_slope=0.2, inplace=True)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).squeeze(4).squeeze(3).squeeze(2)
        x = F.sigmoid(self.linear(x)).squeeze(1)
        return x


class remnet_frame_discriminator(nn.Module):
    def __init__(self):
        super(remnet_frame_discriminator, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (5, 5), (2, 2), (2, 2), bias=True)
        self.conv2d_2 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_3 = nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_4 = nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1), bias=True)
        self.linear = nn.Linear(256, 1)

    # input [out_seq_len, batch_size, 1, 256, 256]
    # output [out_seq_len, batch_size]
    def forward(self, x):
        out_seq_len, batch_size, channels, height, width = x.size()
        mean_x = t.stack([t.mean(t.stack([x[2 * i], x[2 * i + 1]]), dim=0) for i in range(int(out_seq_len/2))], dim=0)
        mean_x = mean_x.reshape(int(out_seq_len/2) * batch_size, channels, height, width)
        mean_x = F.leaky_relu(self.conv2d_1(mean_x), negative_slope=0.2, inplace=True)
        mean_x = F.leaky_relu(self.conv2d_2(mean_x), negative_slope=0.2, inplace=True)
        mean_x = F.leaky_relu(self.conv2d_3(mean_x), negative_slope=0.2, inplace=True)
        mean_x = F.leaky_relu(self.conv2d_4(mean_x), negative_slope=0.2, inplace=True)  # [out_seq_len * batch_size, 256, 16, 16]
        mean_x = F.adaptive_avg_pool2d(mean_x, (1, 1)).squeeze(3).squeeze(2)
        mean_x = F.tanh(self.linear(mean_x)).squeeze(1)
        mean_x = mean_x.reshape(int(out_seq_len/2), batch_size)  # [out_seq_len, batch_size]
        return mean_x


class remnet_frame_patch_discriminator(nn.Module):
    def __init__(self):
        super(remnet_frame_patch_discriminator, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (5, 5), (2, 2), (2, 2), bias=True)
        self.conv2d_2 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_3 = nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_4 = nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_5 = nn.Conv2d(256, 1, (3, 3), (2, 2), (1, 1), bias=True)

    # input [out_seq_len, batch_size, 1, 256, 256]
    # output [out_seq_len, batch_size, 8, 8]
    def forward(self, x):
        out_seq_len, batch_size, channels, height, width = x.size()
        mean_x = t.stack([t.mean(t.stack([x[2 * i], x[2 * i + 1]]), dim=0) for i in range(int(out_seq_len/2))], dim=0)
        mean_x = mean_x.reshape(int(out_seq_len/2) * batch_size, channels, height, width)
        mean_x = F.leaky_relu(self.conv2d_1(mean_x), negative_slope=0.2, inplace=True)
        mean_x = F.leaky_relu(self.conv2d_2(mean_x), negative_slope=0.2, inplace=True)
        mean_x = F.leaky_relu(self.conv2d_3(mean_x), negative_slope=0.2, inplace=True)
        mean_x = F.leaky_relu(self.conv2d_4(mean_x), negative_slope=0.2, inplace=True)  # [out_seq_len * batch_size, 256, 16, 16]
        mean_x = F.tanh(self.conv2d_5(mean_x)).squeeze(1)  # [out_seq_len * batch_size, 8, 8]
        _, height, width = mean_x.size()
        mean_x = mean_x.reshape(int(out_seq_len/2), batch_size, height, width)  # [out_seq_len, batch_size, 8, 8]
        return mean_x
