import torch as t
from torch import nn
import torch.nn.functional as F
from models.RNNs_layers import convlstm
from models.RNNs_layers import convlstm_remnet_cb
from models.RNNs_layers import convlstm_remnet_fb


class echo_lifecycle_encoder(nn.Module):
    def __init__(self):
        super(echo_lifecycle_encoder, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2), bias=True)
        self.conv3d_2 = nn.Conv3d(32, 64, (3, 3, 3), (1, 2, 2), (0, 1, 1), bias=True)
        self.conv3d_3 = nn.Conv3d(64, 128, (3, 3, 3), (1, 2, 2), (1, 1, 1), bias=True)
        self.conv3d_4 = nn.Conv3d(128, 256, (3, 3, 3), (1, 2, 2), (0, 1, 1), bias=True)
        self.linear = nn.Linear(256, 240, bias=True)

    # input [in_seq_len, batch_size, 1, 256, 256]
    # output [batch_size, 240]
    def forward(self, x):
        x = x.permute(1, 2, 0, 3, 4)  # [batch_size, 1, in_seq_len, 256, 256]
        x = F.leaky_relu(self.conv3d_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_3(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv3d_4(x), negative_slope=0.2, inplace=True)  # [batch_size, 256, 1, 16, 16]
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).squeeze(4).squeeze(3).squeeze(2)
        x = F.leaky_relu(self.linear(x), negative_slope=0.2, inplace=True)
        return x


class echo_motion_encoder(nn.Module):
    def __init__(self, in_seq_len):
        super(echo_motion_encoder, self).__init__()
        self.in_seq_len = in_seq_len
        self.conv2d_1 = nn.Conv2d(1, 32, (5, 5), (2, 2), (2, 2), bias=True)
        self.conv2d_2 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_3 = nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_4 = nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1), bias=True)
        self.linear = nn.Linear(256 * (self.in_seq_len - 1), 240, bias=True)

    # input [in_seq_len, batch_size, 1, 256, 256]
    # output [batch_size, 240]
    def forward(self, x):
        batch_size = x.size()[1]
        diff_x = t.cat([x[i + 1] - x[i] for i in range(self.in_seq_len - 1)], dim=0)  # [batch_size * (in_seq_len - 1), 1, 256, 256]
        diff_x = F.leaky_relu(self.conv2d_1(diff_x), negative_slope=0.2, inplace=True)
        diff_x = F.leaky_relu(self.conv2d_2(diff_x), negative_slope=0.2, inplace=True)
        diff_x = F.leaky_relu(self.conv2d_3(diff_x), negative_slope=0.2, inplace=True)
        diff_x = F.leaky_relu(self.conv2d_4(diff_x), negative_slope=0.2, inplace=True)  # [batch_size * (in_seq_len - 1), 256, 16, 16]
        diff_x = F.adaptive_avg_pool2d(diff_x, (1, 1)).squeeze()  # [batch_size * (in_seq_len - 1), 256]
        diff_x = diff_x.reshape(self.in_seq_len - 1, batch_size, -1)
        diff_x = diff_x.permute(1, 0, 2).reshape(batch_size, -1)  # [batch_size, 256 * (in_seq_len - 1)]
        diff_x = F.leaky_relu(self.linear(diff_x), negative_slope=0.2, inplace=True)
        return diff_x


class query_vector_generator(nn.Module):
    def __init__(self):
        super(query_vector_generator, self).__init__()
        self.linear1 = nn.Linear(240, 240, bias=True)
        self.linear2 = nn.Linear(240, 240, bias=True)
        self.linear3 = nn.Linear(240, 240, bias=True)

    # input x1 [batch_size, 240] x2 [batch_size, 240]
    # output [batch_size, 240]
    def forward(self, x1, x2):
        x1 = F.leaky_relu(self.linear1(x1), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.linear2(x2), negative_slope=0.2, inplace=True)
        x = F.tanh(self.linear3(x1 + x2))
        return x


class perception_attention_mechanism(nn.Module):
    def __init__(self):
        super(perception_attention_mechanism, self).__init__()

    # input memory_pool [60, 240, 32, 32] query_vector [batch_size, 240]
    # output [batch_size, 240, 32, 32]
    def forward(self, memory_pool, query_vector):
        memory_pool_averaged = F.adaptive_avg_pool2d(memory_pool, (1, 1)).squeeze()  # [60, 240]
        recalled_memory_features = []
        batch_size = query_vector.size()[0]
        for i in range(batch_size):
            attention_weight = t.cosine_similarity(query_vector[i].repeat(60, 1), memory_pool_averaged, dim=1)
            attention_weight = t.unsqueeze(t.unsqueeze(t.unsqueeze(F.softmax(attention_weight, dim=0), dim=1), dim=2),
                                           dim=3)
            recalled_memory_feature = t.sum(t.mul(memory_pool, attention_weight), dim=0)
            recalled_memory_features.append(recalled_memory_feature)
        recalled_memory_features = t.stack(recalled_memory_features, dim=0)
        return recalled_memory_features


class frame_encoder(nn.Module):
    def __init__(self):
        super(frame_encoder, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (5, 5), (2, 2), (2, 2), bias=True)
        self.conv2d_2 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=True)
        self.conv2d_3 = nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1), bias=True)
        self.conv2d_4 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=True)

    # input [in_seq_len, batch_size, 1, 256, 256]
    # output [in_seq_len, batch_size, 64, 64, 64]
    def forward(self, x):
        in_seq_len, batch_size, channels, height, width = x.size()
        x = x.reshape(in_seq_len * batch_size, channels, height, width)
        x = F.leaky_relu(self.conv2d_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv2d_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv2d_3(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.conv2d_4(x), negative_slope=0.2, inplace=True)
        _, channels, height, width = x.size()
        x = x.reshape(in_seq_len, batch_size, channels, height, width)
        return x


class coarse_branch(nn.Module):
    def __init__(self, out_seq_len, use_gpu=True):
        super(coarse_branch, self).__init__()
        self.out_seq_len = out_seq_len
        self.use_gpu = use_gpu
        self.RNNs_encoder = convlstm(64, 128, 3, 1, use_gpu=self.use_gpu)
        self.RNNs_predictor = convlstm_remnet_cb(64, 128, 3, 1, self.out_seq_len, return_all_layers=False,
                                               use_gpu=self.use_gpu)
        self.spatial_up_sampling = nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1), bias=True)

    def zero_RNNs_predictor_input(self, batch_size, channels, height, width):
        predictor_input = t.zeros([self.out_seq_len, batch_size, channels, height, width])
        if self.use_gpu:
            predictor_input = predictor_input.cuda()
        return predictor_input

    def forward(self, x, recalled_memory_features):
        # Spatial Downsampling
        x = F.avg_pool3d(x, (1, 2, 2), (1, 2, 2))
        # RNNs encoding
        _, encoded_states = self.RNNs_encoder(x)
        # RNNs predicting
        _, batch_size, channels, height, width = x.size()
        predictor_input = self.zero_RNNs_predictor_input(batch_size, channels, height, width)
        predicted_layers_hidden_states, _ = self.RNNs_predictor(predictor_input, recalled_memory_features,
                                                                encoded_states)
        # Spatial Upsampling
        seq_len, _, channels, height, width = predicted_layers_hidden_states.size()
        predicted_layers_hidden_states = predicted_layers_hidden_states.reshape(seq_len * batch_size, channels, height,
                                                                                width)
        predicted_layers_hidden_states = F.leaky_relu(self.spatial_up_sampling(predicted_layers_hidden_states),
                                                      negative_slope=0.2, inplace=True)
        _, channels, height, width = predicted_layers_hidden_states.size()
        predicted_layers_hidden_states = predicted_layers_hidden_states.reshape(seq_len, batch_size, channels, height,
                                                                                width)
        return predicted_layers_hidden_states


class fine_branch(nn.Module):
    def __init__(self, out_seq_len, use_gpu=True):
        super(fine_branch, self).__init__()
        # Temporal Downsampling
        self.out_seq_len = int(out_seq_len / 2)
        self.use_gpu = use_gpu
        self.RNNs_encoder = convlstm(64, 128, 3, 1, use_gpu=self.use_gpu)
        self.RNNs_predictor = convlstm_remnet_fb(64, 128, 3, 1, self.out_seq_len, return_all_layers=False,
                                               use_gpu=self.use_gpu)

    def zero_RNNs_predictor_input(self, batch_size, channels, height, width):
        predictor_input = t.zeros([self.out_seq_len, batch_size, channels, height, width])
        if self.use_gpu:
            predictor_input = predictor_input.cuda()
        return predictor_input

    def forward(self, x, recalled_memory_features):
        # RNNs encoding
        _, encoded_states = self.RNNs_encoder(x)
        # RNNs predicting
        _, batch_size, channels, height, width = x.size()
        predictor_input = self.zero_RNNs_predictor_input(batch_size, channels, height, width)
        predicted_layers_hidden_states, _ = self.RNNs_predictor(predictor_input, recalled_memory_features,
                                                                encoded_states)
        # Temporal Upsampling
        fb_output_feature_sequence = []
        for i in range(self.out_seq_len):
            fb_output_feature_sequence.append(predicted_layers_hidden_states[i].repeat(2, 1, 1, 1, 1))
        fb_output_feature_sequence = t.cat(fb_output_feature_sequence, dim=0)
        return fb_output_feature_sequence


class frame_decoder(nn.Module):
    def __init__(self):
        super(frame_decoder, self).__init__()
        self.deconv2d_1 = nn.ConvTranspose2d(256, 64, (4, 4), (2, 2), (1, 1), bias=True)
        self.deconv2d_2 = nn.ConvTranspose2d(64, 32, (4, 4), (2, 2), (1, 1), bias=True)
        self.deconv2d_3 = nn.ConvTranspose2d(32, 32, (3, 3), (1, 1), (1, 1), bias=True)
        self.deconv2d_4 = nn.ConvTranspose2d(32, 1, (1, 1), (1, 1), bias=True)

    # input [out_seq_len, batch_size, 256, 64, 64]
    # output [out_seq_len, batch_size, 1, 256, 256]
    def forward(self, x):
        out_seq_len, batch_size, channels, height, width = x.size()
        x = x.reshape(out_seq_len * batch_size, channels, height, width)
        x = F.leaky_relu(self.deconv2d_1(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.deconv2d_2(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.deconv2d_3(x), negative_slope=0.2, inplace=True)
        x = F.sigmoid(self.deconv2d_4(x))
        _, channels, height, width = x.size()
        x = x.reshape(out_seq_len, batch_size, channels, height, width)
        return x


class remnet(nn.Module):
    def __init__(self, in_seq_len, out_seq_len, use_gpu=True):
        super(remnet, self).__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.use_gpu = use_gpu
        self.echo_lifecycle_encoder = echo_lifecycle_encoder()
        self.echo_motion_encoder = echo_motion_encoder(self.in_seq_len)
        self.query_vector_generator = query_vector_generator()
        self.lerm_memory_pool = nn.Parameter(t.randn(60, 240, 32, 32))
        self.perception_attention_mechanism = perception_attention_mechanism()
        self.frame_encoder = frame_encoder()
        self.coarse_branch = coarse_branch(self.out_seq_len, self.use_gpu)
        self.fine_branch = fine_branch(self.out_seq_len, self.use_gpu)
        self.frame_decoder = frame_decoder()

    def forward(self, input):
        echo_motion_feature = self.echo_motion_encoder(input)
        echo_lifecycle_feature = self.echo_lifecycle_encoder(input)
        lerm_query_vector = self.query_vector_generator(echo_motion_feature, echo_lifecycle_feature)
        recalled_memory_features = self.perception_attention_mechanism(self.lerm_memory_pool, lerm_query_vector)  # [batchsize, 240, 32, 32]
        encoded_echo_frames = self.frame_encoder(input)
        cb_output_feature_sequence = self.coarse_branch(encoded_echo_frames, recalled_memory_features)
        fb_output_feature_sequence = self.fine_branch(encoded_echo_frames, recalled_memory_features)
        extrapolation_echo_sequence = self.frame_decoder(
            t.cat([cb_output_feature_sequence, fb_output_feature_sequence], dim=2))
        return extrapolation_echo_sequence
