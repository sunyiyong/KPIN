import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat                                # 读取.mat文件
import datetime


# pre-defined neural networks (MLP)
class NeuralNet(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, residual=False, activation="relu"):
    super().__init__()
    if activation == "relu":
      self.activation = torch.relu
    elif activation == "tanh":
      self.activation = torch.tanh
    self.residual = residual
    self.output_dim = output_dim
    self.layer1 = nn.Linear(input_dim, hidden_dim[0])
    self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[1])
    self.layer3 = nn.Linear(hidden_dim[1], hidden_dim[2])
    self.layer4 = nn.Linear(hidden_dim[2], output_dim)

  def forward(self, x):
    res = self.activation(self.layer1(x))
    res = self.activation(self.layer2(res))
    res = self.activation(self.layer3(res))
    res = self.layer4(res)
    if self.residual:
      out = res + x
    else:
      out = res
    return out

class ChannelDataset(data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)


def split_dataset_Informer(global_vars):
    """
        :param global_vars:

            global_vars.init_y: np, complex-valued, y_dim x sequence_len
            global_vars.init_h: bp, complex-valued, h_dim x sequence_len

        :param lag:

    :return:
        informer_train_input: Torch.tensor
        informer_train_target: Torch.tensor
        informer_test_input: Torch.tensor
        informer_test_target: Torch.tensor
    """

    dev  = global_vars.init_dev

    y_np = global_vars.init_y    # y_dim x sequence_len
    h_np = global_vars.init_h    # h_dim x sequence_len

    I_order = global_vars.I_order
    O_order = global_vars.O_order
    n_train = global_vars.informer_train_len - (I_order+O_order) + 1
    delta_t = global_vars.informer_test_start - global_vars.informer_train_len

    seq_len = y_np.shape[-1]

    y_np_real = y_np.real   # y_dim x sequence_len
    y_np_imag = y_np.imag   # y_dim x sequence_len
    y_real = torch.from_numpy(y_np_real)
    y_imag = torch.from_numpy(y_np_imag)
    y = torch.cat((y_real, y_imag), dim=0) # (2 * y_dim) x sequence_len


    h_np_real = h_np.real   # h_dim x sequence_len
    h_np_imag = h_np.imag   # h_dim x sequence_len
    h_real = torch.from_numpy(h_np_real)
    h_imag = torch.from_numpy(h_np_imag)
    h = torch.cat((h_real, h_imag), dim=0)  # (2 * h_dim) x sequence_len

    # wrap data into regular input-output pairs
    input_dim_ = y.shape[0]
    output_dim = h.shape[0]

    # kformer_GRU_in_dim = I_order * input_dim_
    inputs = torch.empty(seq_len - I_order, I_order, input_dim_)   # (seq_len - I_order) x kformer_GRU_in_dim
    outputs = torch.empty(seq_len - I_order,O_order, output_dim) # (seq_len - I_order) x output_dim
    for t in range(seq_len - I_order - O_order):
        # print(init_f"t: {t},    t+I_order: {t+I_order} ")
        inputs[t, :, :] = y[:, t:t+I_order].t().unsqueeze(0)
        outputs[t, :, :] = h[:, t+I_order:t+I_order+O_order].t().unsqueeze(0)

    n_test_start = n_train + O_order - 1

    global_vars.informer_train_input  = inputs[ delta_t:delta_t+n_train, : , : ].to(dev)
    global_vars.informer_train_target = outputs[ delta_t:delta_t+n_train, : , : ].to(dev)

    global_vars.informer_test_input   = inputs[ delta_t+n_test_start :, : , : ].to(dev)
    global_vars.informer_test_target  = outputs[ delta_t+n_test_start:, : , : ].to(dev)  # 2024.4.3改为p=4，L=1000

    # informer_train_input  = global_vars.informer_train_input.cpu().detach().numpy().squeeze()
    # informer_train_target = global_vars.informer_train_target.cpu().detach().numpy().squeeze()
    # informer_test_input   = global_vars.informer_test_input.cpu().detach().numpy().squeeze()
    # informer_test_target  = global_vars.informer_test_target.cpu().detach().numpy().squeeze()


    return global_vars.informer_train_input, global_vars.informer_train_target, global_vars.informer_test_input, global_vars.informer_test_target


def data_normalization(X_train, Y_train, X_test, Y_test):
    mu_x = torch.mean(X_train,dim=0).to(dev)  # kformer_GRU_in_dim
    std_x = torch.std(X_train, dim=0).to(dev) # kformer_GRU_in_dim

    mu_y = torch.mean(Y_train,dim=0).to(dev)  # output_dim
    std_y = torch.std(Y_train, dim=0).to(dev) # output_dim


    X_train_normal = (X_train - mu_x)/std_x
    Y_train_normal = (Y_train - mu_y)/std_y

    X_test_normal = (X_test - mu_x)/std_x
    Y_test_normal = (Y_test - mu_y)/std_y

    return X_train_normal, Y_train_normal, X_test_normal, Y_test_normal, mu_x, std_x, mu_y, std_y


def data_recover(mu, std, data):
    x_recover = data * std + mu
    return x_recover















