import torch
import torch.nn as nn
from casCN_model.ChebyGCN_Direc import ChebConv
import numpy as np

class chebLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ChebLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of features of input tensor.
        hidden_dim: int
            Number of channels of hidden state. LSTM神经元的数目
        kernel_size: (int)
            Size of the convolutional kernel.GCN的卷积核的阶数
        bias: bool
            Whether or not to add the bias.
        """

        super(chebLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.bias = bias

        self.chebgcn1 = ChebConv(self.input_dim, 4 * self.hidden_dim, K = self.kernel_size)
        self.chebgcn2 = ChebConv(self.hidden_dim, 4 * self.hidden_dim, K= self.kernel_size)


    # 这里forward有两个输入参数，input_tensor 是一个4维数据c
    # （t时间步,b输入batch_ize,c输出数据通道数--维度,h,w图像高乘宽）
    # hidden_state=None 默认刚输入hidden_state为空，等着后面给初始化
    def forward(self, input_tensor, graph, cur_state):
        h_cur, c_cur = cur_state

        x = self.chebgcn1(input_tensor, graph)
        h_cur_conv = self.chebgcn2(h_cur,graph)

        combined = x + h_cur_conv
        #combined = torch.cat([x, h_cur], dim=1)
          # concatenate along channel
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_dim, dim=2) #将combined的第1维 划分为hidden_dim块
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, input_dim, hidden_dim):
        return (torch.zeros(batch_size, input_dim, hidden_dim,device=self.chebgcn1.weight.device),
                torch.zeros(batch_size, input_dim, hidden_dim, device=self.chebgcn1.weight.device))


class chebLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of feature in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(chebLSTM, self).__init__()

        # self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        print(num_layers)
        kernel_size = self._extend_for_multilayer(kernel_size,  num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim,  num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(chebLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, graph, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            4-D Tensor either of shape (t, b,  h, w) or (b, t, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        # 先调整一下输出数据的排列
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3)

        # 取出图片的数据，供下面初始化使用
        b, _, h, w = input_tensor.shape

        # Implement stateful ConvLSTM #初始化hidd_state,利用后面和lstm单元中的初始化函数
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(b, self.input_dim, self.hidden_dim)

        # 储存输出数据的列表
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.shape[1]
        # 初始化输入数据
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            output_inner_list = []
            layer_output_1 = []
            # 每一个时间步都更新 h,c
            # 注意这里self.cell_list是一个模块(容器)
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :], graph=graph[t],
                                                 cur_state=[h, c])

                # 储存输出，注意这里 h 就是此时间步的输出
                output_inner.append(h)
                output_inner_list.append(h.tolist())

            # 这一层的输出作为下一次层的输入
            layer_output = torch.stack(output_inner, dim=1)
            layer_output_1.append(output_inner_list)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output_1)
            # 储存每一层的状态h，c
            last_state_list.append([h, c])

        # 选择要输出所有数据，还是输出最后一层的数据
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, input_dim, hidden_dim):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size,input_dim, hidden_dim[0]))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = param * num_layers
        return param

class time_Decay(nn.Module):
    def __init__(self, num_layers, time_interal_n):
        super(time_Decay,self).__init__()

        self.num_layers = num_layers

        self.time_weight = nn.Parameter(torch.FloatTensor(time_interal_n,1))

    def forward(self, last_h, n_step, hidden_dim, rnn_index, time_interval_index):
        for t in range(self.num_layers):
            last_h = last_h[t]
            last_h = torch.tensor(last_h)
            last_h = last_h.squeeze().permute(1, 0, 2, 3)  # batch first
            last_h = torch.sum(last_h, dim=2)
            last_h = torch.reshape(last_h, (-1, hidden_dim[t]))
            rnn_index = torch.reshape(rnn_index, (-1, 1))
            last_h = torch.mul(rnn_index, last_h)

            time_interval_index = torch.reshape(time_interval_index, (-1, 6))
            time_interval_index = time_interval_index.matmul(self.time_weight)
            last_h = torch.mul(time_interval_index, last_h)
            last_h = torch.reshape(last_h, (-1, n_step, hidden_dim[t]))
            last_h = torch.sum(last_h, dim=1)

        return last_h

class MODEL(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first, n_time_interval, dense1, dense2):
        super(MODEL, self).__init__()

        self.hidden_dim = hidden_dim
        self.dense1 = dense1
        self.dense2 = dense2

        self.chebylstm = chebLSTM(input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first)
        self.time_decay = time_Decay(num_layers,n_time_interval)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim[-1], self.dense1),
            nn.Tanh(),
            nn.Linear(self.dense1, self.dense2),
            nn.Tanh(),
            nn.Linear(self.dense2,1),
            nn.Tanh()
        )
    def forward(self, input_tensor, graph, n_step,
                hidden_dim, rnn_index, time_interval_index):

        last_h,_ = self.chebylstm(input_tensor, graph)
        last_h = self.time_decay(last_h, n_step, hidden_dim, rnn_index, time_interval_index)
        pred = self.mlp(last_h)
        return pred




