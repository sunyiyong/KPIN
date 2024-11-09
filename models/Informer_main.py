import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat                                     # 用于读取.mat文件里的struct结构体
import datetime
import time                                                      # 返回时间以s为单位
import os


from MLP_utils import NeuralNet, split_dataset_MLP, data_normalize_MLP, data_recover_MLP
from Informer_utils import split_dataset_Informer
from Informer_model.model import Informer, InformerStack, InformerStack_e2e
from Informer_model.metrics import NMSELoss, Adap_NMSELoss


####################
### Informer model
####################

def Informer_pipeline(global_vars):

    # data loading
    informer_train_input  = global_vars.informer_train_input
    informer_train_target = global_vars.informer_train_target
    informer_test_input   = global_vars.informer_test_input
    informer_test_target  = global_vars.informer_test_target
    dev                   = global_vars.init_dev
    epsilon               = global_vars.informer_epsilon
    num_epochs            = global_vars.informer_N_epochs

    model_dim             = global_vars.informer_model_dim  # 这里不能太大，否则inference的时候会出问题，yiyong-0820
    num_head              = global_vars.informer_num_head  # num of heads
    # data normalization
    informer_train_input, informer_train_target, informer_test_input, informer_test_target, mu_x_informer, std_x_informer, mu_y_informer, std_y_informer = data_normalize_MLP(informer_train_input, informer_train_target, informer_test_input, informer_test_target, dev)

    # 转移到gpu上
    informer_train_input = informer_train_input.to(dev)       # n_train x input_dim
    informer_train_target = informer_train_target.to(dev)       # n_train x output_dim

    informer_test_input = informer_test_input.to(dev)     # n_test x input_dim
    informer_test_target = informer_test_target.to(dev)     # n_test x output_dim

    # model结构准备
    num_train = informer_train_input.shape[0]
    seq_len = informer_train_input.shape[1]
    label_len = 10
    pred_len = informer_train_target.shape[1]

    enc_in = informer_train_input.shape[2]      #encoder input size
    dec_in =  informer_train_input.shape[2]     #decoder input size
    c_out = informer_train_input.shape[2]       #output size
    # num_head = 8                             #num of heads
    e_layers = 4                            #num of encoder layers
    d_layers = 3                            #num of decoder layers
    d_ff = 64                               #dimension of fcn
    factor = 5                              #probsparse attn factor
    distil = True                           #whether to use distilling in encoder
    dropout = 0.05                          #dropout
    attn = 'full'                           #attention used in encoder, options:[prob, full]
    embed = 'fixed'                         #time features encoding, options:[timeF, fixed, learned]
    activation = 'gelu'                     #activation
    output_attention = False                #whether to output attention in ecoder
    Informer_mode = "InformerStack"         #options:[InformerStack, InformerStack_e2e]

    informer_settings_e2e = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}'.format('informerstack_e2e',
                    seq_len, label_len, pred_len,
                    64, num_head, e_layers, d_layers, d_ff, attn, factor, embed, distil)

    informer_settings = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}'.format('informerstack',
                    seq_len, label_len, pred_len,
                    64, num_head, e_layers, d_layers, d_ff, attn, factor, embed, distil)

    print(informer_settings)
    print(informer_settings_e2e)

    if Informer_mode == "InformerStack":
        informer = InformerStack(
            enc_in,
            dec_in,
            c_out,
            seq_len,
            label_len,
            pred_len,
            factor,
            model_dim,
            num_head,
            e_layers,
            d_layers,
            d_ff,
            dropout,
            attn,
            embed,
            activation,
            output_attention,
            distil,
            dev
        ).to(dev)
    elif Informer_mode == "InformerStack_e2e":
        informer = InformerStack_e2e(
            enc_in,
            dec_in,
            c_out,
            seq_len,
            label_len,
            pred_len,
            factor,
            model_dim,
            num_head,
            e_layers,
            d_layers,
            d_ff,
            dropout,
            attn,
            embed,
            activation,
            output_attention,
            distil,
            dev
        ).to(dev)

    # loss function
    criterion = torch.nn.MSELoss()

    enc_inp = informer_train_input
    dec_inp =  torch.zeros_like( informer_train_input[:, -pred_len:, :] ).to(dev)
    dec_inp =  torch.cat([enc_inp[:, seq_len - label_len:seq_len, :], dec_inp], dim=1)

    enc_inp_test = informer_test_input
    dec_inp_test =  torch.zeros_like( informer_test_input[:, -pred_len:, :] ).to(dev)
    dec_inp_test =  torch.cat([enc_inp_test[:, seq_len - label_len:seq_len, :], dec_inp_test], dim=1)

    # training parameters
    batch_size = 128                                        # batch size
    num_epoch_Informer = global_vars.informer_N_epochs      # number of epoch
    num_iter = int(num_train/batch_size)
    lr = 0.0005
    optimizer = torch.optim.Adam(informer.parameters(), lr=lr)

    #########################
    ## 训练Informer
    #########################
    # informer.nmse_informer_epoch = torch.zeros([num_epoch_Informer])  # 记录每个epoch的loss
    # informer.time_informer_epoch = torch.zeros([num_epoch_Informer])  # 记录每个epoch的训练时间
    informer.nmse_informer_epoch = []  # 记录每个epoch的loss
    informer.time_informer_epoch = []  # 记录每个epoch的训练时间
    losses_informer = []
    time_Informer_train = 0.0


    epoch = 0
    # while informer.nmse_informer_epoch[epoch] >= epsilon and epoch < num_epoch_Informer:
    while True and epoch < num_epochs:
    # for epoch in range(num_epoch_Informer):

        start_informer = time.time()

        indices = torch.randperm(num_train)
        X_train_rand = enc_inp[indices, :, :]
        Y_train_rand = informer_train_target[indices, :, :]
        dec_inp_rand = dec_inp[indices, :, :]
        informer.train()
        MSE_all_informer = torch.empty(num_iter)
        for t in range(num_iter):
            # forward
            Y_informer_output = informer(X_train_rand[t*batch_size:(t+1)*batch_size,:,:], dec_inp_rand[t*batch_size:(t+1)*batch_size,:,:])

            # loss
            MSE_informer = criterion(Y_informer_output, Y_train_rand[t*batch_size:(t+1)*batch_size,:,:])
            MSE_all_informer[t] = MSE_informer.item()

            # backward
            optimizer.zero_grad()
            MSE_informer.backward()
            optimizer.step()
        losses_informer.append(MSE_all_informer.mean())

        end_informer = time.time()
        t_Informer_epoch = end_informer - start_informer
        time_Informer_train = time_Informer_train + t_Informer_epoch

        informer.nmse_informer_epoch.append(MSE_all_informer.mean())   # 存储本次epoch的nmse
        informer.time_informer_epoch.append(t_Informer_epoch)  # 存储本次epoch的训练时间


        # informer.nmse_informer_epoch[epoch] = MSE_all_informer.mean()   # 存储本次epoch的nmse
        # informer.time_informer_epoch[epoch] = t_Informer_epoch          # 存储本次epoch的训练时间
        print( "Informer Epoch: {}, Loss:  {:.4f}, time: {:.2f}s".format(epoch, informer.nmse_informer_epoch[epoch] ,informer.time_informer_epoch[epoch]) )

        if epoch > 0 and losses_informer[epoch] < epsilon:
            break  # 如果损失小于epsilon，结束训练

        epoch += 1
        torch.cuda.empty_cache()  # 清空缓存，init_N=64推理会爆炸，yiyong-0731

    global_vars.informer_N_epochs_actual = epoch
    #########################
    ## 测试Informer
    #########################

    global_vars.informer_N_epochs_actual = epoch
    start_Informer_test = time.time()

    torch.cuda.empty_cache()  # 清空缓存，init_N=64推理会爆炸，yiyong-0731

    informer.eval()
    loss_informer = torch.nn.MSELoss(reduction='none')
    num_test = informer_test_input.shape[0]                  # n_test
    Y_pred_informer = informer(enc_inp_test, dec_inp_test)   # n_test x output_dim

    # data recovering
    Y_pred_informer = data_recover_MLP(mu=mu_y_informer, std=std_y_informer, data=Y_pred_informer)
    informer_test_target = data_recover_MLP(mu=mu_y_informer, std=std_y_informer, data=informer_test_target)

    # computing criterion
    diff_square_informer = loss_informer(Y_pred_informer, informer_test_target)       # n_test x output_dim
    norm_square_informer = diff_square_informer.sum(dim=-1)    # n_test

    norm_y_test_square_informer = torch.norm(informer_test_target, dim=-1) ** 2

    nmse_informer      = norm_square_informer / norm_y_test_square_informer
    nmse_informer_cpu  = nmse_informer.detach().cpu().numpy()
    nmse_informer_dB   = 10 * np.log10( nmse_informer_cpu )         # 转成dB格式

    end_Informer_test  = time.time()
    time_Informer_test = end_Informer_test - start_Informer_test

    global_vars.Informer                    = informer
    global_vars.Informer.nmse_informer      = nmse_informer
    global_vars.Informer.Y_pred_informer    = Y_pred_informer
    global_vars.Informer.time_informer_test = time_Informer_test

    # print( "Training Time for Informer {} epochs: {:.2f}s".format(global_vars.informer_N_epochs, time_Informer_train))
    # print("Testing Time for Informer: {:.2f}s ".format(time_Informer_test) )
    # print("Total Time for Informer: {:.2f}s ".format(time_Informer_train + time_Informer_test) )





