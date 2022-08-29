import torch as t
from torch import nn
from models.RNNs_units import convlstm_unit


class convlstm(nn.Module):
    # hidden_channels should be a list if the layers_num >= 2
    def __init__(self, in_channels, hidden_channels, kernel_size, layers_num, bias=True, return_all_layers=True, use_gpu=True):
        super(convlstm, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.layers_num = layers_num
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.use_gpu = use_gpu
        units_list = []
        for i in range(layers_num):
            cur_in_channels = self.in_channels if i == 0 else self.hidden_channels[i-1]
            cur_hidden_channels = self.hidden_channels if layers_num == 1 else self.hidden_channels[i]
            units_list.append(convlstm_unit(cur_in_channels, cur_hidden_channels, self.kernel_size, bias=self.bias))
        self.units_list = nn.ModuleList(units_list)

    def zero_ini_layers_states(self, batch_size, height, width):
        ini_layers_states = []
        for i in range(self.layers_num):
            cur_hidden_channels = self.hidden_channels if self.layers_num == 1 else self.hidden_channels[i]
            zero_state = t.zeros([batch_size, cur_hidden_channels, height, width])
            if self.use_gpu:
                zero_state = zero_state.cuda()
            zero_layer_states = (zero_state, zero_state)
            ini_layers_states.append(zero_layer_states)
        return ini_layers_states

    # input:
    # input [seq_len, batch_size, in_channels, height, width]
    # ini_layers_states [(h_1, c_1), (h_2, c_2), ..., (h_l, c_l)]
    # output:
    # layers_hidden_states [[seq_len, batch_size, hidden_channels, height, width],...]
    # layers_states [(h_1, c_1), (h_2, c_2), ..., (h_l, c_l)]
    # h c [batch_size, hidden_channels, height, width]
    def forward(self, input, ini_layers_states=None):
        seq_len, batch_size, _, height, width = input.size()
        if ini_layers_states is None:
            ini_layers_states = self.zero_ini_layers_states(batch_size, height, width)
        layers_hidden_states = []
        layers_states = []
        cur_input = input
        for layer_index in range(self.layers_num):
            state = ini_layers_states[layer_index]
            layer_hidden_states = []
            for step in range(seq_len):
                state = self.units_list[layer_index](cur_input[step, :, :, :, :], state)
                layer_hidden_states.append(state[0])
            layers_states.append(state)
            layer_hidden_states = t.stack(layer_hidden_states, dim=0)
            layers_hidden_states.append(layer_hidden_states)
            cur_input = layer_hidden_states
        if self.return_all_layers:
            return layers_hidden_states, layers_states
        else:
            return layers_hidden_states[-1], layers_states[-1]
