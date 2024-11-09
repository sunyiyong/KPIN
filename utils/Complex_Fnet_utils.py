import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np



########################################
# F-net 网络结构, yiyong2023.11.7逐行运行检查没有问题。
########################################
class Fnet(nn.Module):
  def __init__( self, global_vars ):
    super().__init__()

    self.dev               = global_vars.init_dev
    ########################################
    # Fnet 维度计算，固定Gnet训练Fnet，Fnet输入是x输出是h——yiyong.2023.11.19
    ########################################
    self.FC1_in_dim        = 2 * global_vars.m  # 2 * MN，
    self.GRU_in_dim        = global_vars.Fnet_GRU_in_dim  # 自定义
    self.GRU_hn_dim        = global_vars.Fnet_GRU_hn_dim  # 自定义
    self.GRU_num_layers    = 1
    self.GRU_num_direction = 1
    self.GRU_batch_size    = global_vars.Fnet_pretrain_batchsize  # 在ssm结构中，每次只输入一个batch
    self.GRU_seq_len       = global_vars.p  # p                 = 4
    self.FC2_out_dim       = global_vars.Fnet_FC2_out_dim
    self.FC3_out_dim       = 2 * global_vars.m  # 2 *　MN


    ########################################
    # Fnet 由FC1, GRU, FC2, FC3组成
    ########################################
    # FC1
    self.l1_linear = nn.Linear( self.FC1_in_dim, self.GRU_in_dim, bias=True ).to(self.dev)    # 2n x in_dim
    self.l1_relu   = nn.ReLU()

    # GRU
    self.Fnet_GRU  = nn.GRU(self.GRU_in_dim, self.GRU_hn_dim, self.GRU_num_layers, batch_first=True).to(self.dev)   # 这里无需指定seq_len，因为该变量决定GRU展开几次，不影响单个GRU单元的结构，yiyong-2023.11.7
    # 在batch_first=True情况下，GRU_input维度(GRU_batch_size, GRU_seq_len, GRU_in_dim), yiyong-2023.11.6
    self.hn        = torch.zeros( self.GRU_num_direction * self.GRU_num_layers, self.GRU_batch_size, self.GRU_hn_dim).to(self.dev)
    # (1, 996, hn_dim), 初始化hidden_cell, yiyong-2023.11.6

    # FC2
    self.l2_linear = nn.Linear( self.GRU_hn_dim, self.FC2_out_dim, bias=True ).to(self.dev)   # hn_dim x FC2_out_dim
    self.l2_relu   = nn.ReLU()

    # FC3, no activation
    self.l3_linear = nn.Linear( self.FC2_out_dim, self.FC3_out_dim, bias=True ).to(self.dev)  # FC2_out_dim x 2m


  ########################################
  # F-net 前向传播, yiyong2023.11.9逐行运行检查没有问题。
  ########################################
  def Fnet_pretrain_forward( self, pretrain_input, hn ):
      # 输出量级是e-2，注意！！！！！！

      hn = hn

      # 执行FC1: (996, 4, 2n) -> (996, 4, GRU_in_dim)
      l1_out          = self.l1_linear( pretrain_input )    # (996, 4, GRU_in_dim)
      la1_out         = self.l1_relu( l1_out )              # (996, 4, GRU_in_dim)


      # 执行GRU: (996, 4, GRU_in_dim) -> (996, self.GRU_hn_dim)
      GRU_out, _      = self.Fnet_GRU( la1_out, hn )         # GRU_out维度[996, 4, self.GRU_hn_dim]
      GRU_out_reshape = GRU_out[:, -1, :].squeeze()               # 输出维度[996, self.GRU_hn_dim]


      # 执行FC2: (996, self.GRU_hn_dim) -> (996, FC2_out_dim)
      l2_out          = self.l2_linear( GRU_out_reshape )       # (996, FC2_out_dim)
      la2_out         = self.l2_relu( l2_out )                  # (996, FC2_out_dim)


      # 执行FC3: (996, FC2_out_dim) -> (996, 2m), no activation
      l3_out          = self.l3_linear( la2_out )              # (996, 2m)

      return l3_out




################################################################################
# Data Pre-processing 数据预处理，输入h真值输出h真值, yiyong2023.11.21逐行运行检查没有问题。
################################################################################
def split_dataset_Fnet( global_vars ):

    ##################################################
    # 提取global_vars里的参数，yiyong-2023.11.5
    ##################################################
    dev                                 = global_vars.init_dev
    y_np                                = global_vars.init_y       # ndarray, 带噪声观测值
    # y_true_np                           = global_vars.init_y_true  # ndarray, 无噪声观测值
    # noise_np                            = global_vars.init_noise   # ndarray，观测噪声, tau*N x 4000
    h_np                                = global_vars.init_h       # ndarray, M*N  x 4000
    I_order                             = global_vars.Fnet_p
    O_order                             = 1
    # 数据索引解耦，训练和测试不一定连在一起-yiyong-2023.11.5
    pretrain_start                      = global_vars.Fnet_pretrain_start    # 1048
    pretrain_len                        = global_vars.Fnet_pretrain_len      # 1000
    pretrain_end                        = global_vars.Fnet_pretrain_end      # 2048
    test_start                          = global_vars.Fnet_pretest_start     # 2048
    test_len                            = global_vars.Fnet_pretest_len       # 100
    test_end                            = global_vars.Fnet_pretest_end       # 2148



    ##################################################
    # 训练集Training Set，yiyong-2023.11.5，注意输入y_1：y_p、标签y_p+1
    ##################################################
    n_pretrain                          = pretrain_len - (I_order+O_order) + 1                 # 996, 每(I_order+O_order)步作为一个样本
    global_vars.Fnet_pretrain_batchsize = n_pretrain
    # 输入和输出都是y，concatenate real and imaginary parts of y
    y_pretrain_np      = y_np[:, pretrain_start: pretrain_end]                # (128,1000),1048:2047
    y_pretrain_np_real = y_pretrain_np.real                                   # ndarray--n x 1000
    y_pretrain_np_imag = y_pretrain_np.imag                                   # ndarra--n x 1000
    y_pretrain_real    = torch.from_numpy( y_pretrain_np_real )               # tensor--n x 1000
    y_pretrain_imag    = torch.from_numpy( y_pretrain_np_imag )               # tensor--n x 1000
    y_pretrain         = torch.cat( (y_pretrain_real, y_pretrain_imag), dim= 0 ) # tensor--2*n x 1000
    # concatenate real and imaginary parts of h
    h_pretrain_np      = h_np[:, pretrain_start: pretrain_end]
    h_pretrain_np_real = h_pretrain_np.real                            # ndarray--h_dim x sequence_len
    h_pretrain_np_imag = h_pretrain_np.imag                            # ndarray--h_dim x sequence_len
    h_pretrain_real    = torch.from_numpy( h_pretrain_np_real )        # tensor--h_dim x sequence_len
    h_pretrain_imag    = torch.from_numpy( h_pretrain_np_imag )        # tensor--h_dim x sequence_len
    h_pretrain         = torch.cat( (h_pretrain_real, h_pretrain_imag), dim= 0 )      # tensor, (2 * h_dim) x sequence_len



    pretrain_input_dim_           = y_pretrain.shape[0]   # 2n
    pretrain_output_dim           = h_pretrain.shape[0]   # 2m
    pretrain_inputs               = torch.zeros( n_pretrain, I_order, pretrain_input_dim_ )  # 拼接后的样本数 * p阶 * 2y维度，和原始数据是转置关系，yiyong-2023.11.5
    pretrain_outputs              = torch.zeros( n_pretrain, O_order, pretrain_output_dim )  # 拼接后的样本数 * 1阶 * 2y维度，和原始数据是转置关系，yiyong-2023.11.5
    # wrap data into regular input-output pairs
    for t in range( n_pretrain ):
        pretrain_inputs[t, :, :]  = y_pretrain[:, t:t+I_order].t().unsqueeze(0)                  # 扩充的维度 * p阶 * 2y维度，和原始数据是转置关系，yiyong-2023.11.5
        pretrain_outputs[t, :, :] = h_pretrain[:, t+I_order:t+I_order+O_order].t().unsqueeze(0)  # 扩充的维度 * 1阶 * 2y维度，和原始数据是转置关系，yiyong-2023.11.5

    # split into training / testing sets, 注意都是y
    global_vars.Fnet_pretrain_input  = pretrain_inputs.to(dev)
    global_vars.Fnet_pretrain_target = pretrain_outputs.to(dev)



    ##################################################
    # 测试集Testing Set，yiyong-2023.11.6，注意输入y_1：y_p、标签h_p+1
    ##################################################
    n_pretest          = test_len                              # 100, 和训练集不同，每一步都要test
    global_vars.Fnet_pretest_batchsize = n_pretest
    # concatenate real and imaginary parts of y
    y_pretest_np       = y_np[:, test_start - I_order: test_end]  # 2044:2147, n x 104, 和训练集不同，需要补充前面p个时刻的数据，yiyong-2023.11.5
    y_pretest_np_real  = y_pretest_np.real                        # ndarray, n x 104
    y_pretest_np_imag  = y_pretest_np.imag                        # ndarray, n x 104
    y_pretest_real     = torch.from_numpy(y_pretest_np_real)      # tensor--n x 104
    y_pretest_imag     = torch.from_numpy(y_pretest_np_imag)      # tensor--n x 104
    y_pretest          = torch.cat( (y_pretest_real, y_pretest_imag), dim=0 )    # tensor, 2n x 104
    # concatenate real and imaginary parts of h
    h_pretest_np       = h_np[:, test_start - I_order: test_end]  # 2044:2147, m x 104, 和训练集不同，需要补充前面p个时刻的数据，yiyong-2023.11.5
    h_pretest_np_real  = h_pretest_np.real                        # ndarray, m x 104
    h_pretest_np_imag  = h_pretest_np.imag                        # ndarray, m x 104
    h_pretest_real     = torch.from_numpy(h_pretest_np_real)      # tensor--m x 104
    h_pretest_imag     = torch.from_numpy(h_pretest_np_imag)      # tensor--m x 104
    h_pretest          = torch.cat( (h_pretest_real, h_pretest_imag), dim=0 )    # tensor, 2m x 104



    pretest_input_dim_ = y_pretest.shape[0]  # 2n
    pretest_output_dim = h_pretest.shape[0]  # 2m
    pretest_inputs     = torch.zeros(n_pretest, I_order, pretest_input_dim_)  # 拼接后的样本数 * p阶 * 2y维度，和原始数据是转置关系，yiyong-2023.11.5
    pretest_outputs    = torch.zeros(n_pretest, O_order, pretest_output_dim)  # 拼接后的样本数 * 1阶 * 2h维度，和原始数据是转置关系，yiyong-2023.11.5
    # wrap data into regular input-output pairs
    for t in range( n_pretest ):
        pretest_inputs[t, :, :]  = y_pretest[:, t:t+I_order].t().unsqueeze(0)                   # 扩充的维度 * p阶 * 2y维度，和原始数据是转置关系，yiyong-2023.11.5
        pretest_outputs[t, :, :] = h_pretest[:, t+I_order:t+I_order+O_order].t().unsqueeze(0)  # 扩充的维度 * 1阶 * 2h维度，和原始数据是转置关系，yiyong-2023.11.5

    # split into testing / testing sets，得把test和test的起始点加进去，做成可以自由调整开始结束的点
    global_vars.Fnet_pretest_input  = pretest_inputs.to(dev)
    global_vars.Fnet_pretest_target = pretest_outputs.to(dev)

    # # 转成numpy格式，方便检查结果是否符合预期
    # Fnet_pretrain_input  = global_vars.Fnet_pretrain_input.cpu().detach().numpy().squeeze()    # 996 x 4 x 2n
    # Fnet_pretrain_target = global_vars.Fnet_pretrain_target.cpu().detach().numpy().squeeze()   # 996 x 2n
    # Fnet_pretest_input   = global_vars.Fnet_pretest_input.cpu().detach().numpy().squeeze()     # 100 x 4 x 2n
    # Fnet_pretest_target  = global_vars.Fnet_pretest_target.cpu().detach().numpy().squeeze()    # 100 x 2m



########################################
# Data Normalize 数据归一化,输入h输出h，input和target的mu与std都很接近， yiyong2023.11.21逐行运行检查没有问题。
########################################
def data_normalize_Fnet( global_vars ):

    dev             = global_vars.init_dev
    pretrain_input  = global_vars.Fnet_pretrain_input  # 996 x 4 x 2n
    pretrain_target = global_vars.Fnet_pretrain_target # 996 x 1 x 2m, 没有squeeze
    pretest_input   = global_vars.Fnet_pretest_input  # 996 x 4 x 2n
    pretest_target  = global_vars.Fnet_pretest_target  # 996 x 1 x 2m, 没有squeeze

    # elemenwisely compute mean and standard deviation based on training data
    # x本身量级1e-6，缩放完在[-1,1]，输出h的量级是1e-6，基本可以实现网络权重在0附近，理论上比normalize想达到的效果更好。
    mu_x    = torch.mean(pretrain_input,dim=0, keepdim=True).to(dev)   # 1 x 4 x 2n
    std_x   = torch.std(pretrain_input, dim=0, keepdim=True).to(dev)   # 1 x 4 x 2n
    # 其实没必要对y缩放，因为y不是网络的直接输出，并且nmse相当于缩放, yiyong-2023.11.7
    mu_y    = torch.mean(pretrain_target,dim=0, keepdim=True).to(dev)   # 1 x 1 x 2n
    std_y   = torch.std(pretrain_target, dim=0, keepdim=True).to(dev)   # 1 x 1 x 2n


    # Normalization
    global_vars.Fnet_pretrain_input_normal  = (pretrain_input - mu_x)/std_x       # [1e-1,1e1]之间, elemenwise division, yiyong-2023.11.5
    global_vars.Fnet_pretrain_target_normal = (pretrain_target - mu_y) / std_y  # elemenwise division, yiyong-2023.11.7
    # # 其实标签没必要处理
    global_vars.Fnet_pretest_input_normal = (pretest_input - mu_x) / std_x
    global_vars.Fnet_pretest_target_normal  = (pretest_target - mu_y)/std_y          # elemenwise division, yiyong-2023.11.7
    # 保存数据
    global_vars.Fnet_pretrain_mu_x  = mu_x
    global_vars.Fnet_pretrain_std_x = std_x
    global_vars.Fnet_pretrain_mu_y  = mu_y
    global_vars.Fnet_pretrain_std_y = std_y


    # # 转成numpy格式，方便检查结果是否符合预期
    # Fnet_pretrain_input_normal  = global_vars.Fnet_pretrain_input_normal.cpu().detach().numpy().squeeze()   # 996 x 4 x 2n
    # Fnet_pretest_input_normal   = global_vars.Fnet_pretest_input_normal.cpu().detach().numpy().squeeze()    # 100 x 4 x 2n
    # Fnet_pretrain_target_normal = global_vars.Fnet_pretrain_target_normal.cpu().detach().numpy().squeeze()
    # Fnet_pretest_target_normal  = global_vars.Fnet_pretest_target_normal.cpu().detach().numpy().squeeze()

    a = 1
    # return pretrain_input_normal, pretrain_target_normal, test_input_normal, test_target_normal, mu_x, std_x, mu_y, std_y


















