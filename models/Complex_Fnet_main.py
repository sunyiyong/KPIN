import torch
import torch.nn as nn
import numpy as np
import datetime
import time                                                      # 返回时间以s为单位
import os


#########################
# Fnet pretrain函数
#########################
def Fnet_pretrain(global_vars):

    dev                     = global_vars.init_dev
    m                       = global_vars.m
    M                       = global_vars.init_M
    N                       = global_vars.init_N
    tau                     = global_vars.init_tau
    rho                     = global_vars.init_rho
    Fnet_pretrain_input     = global_vars.Fnet_pretrain_input_normal.squeeze()    # 996 x 4 x 2m_标签无需缩放, yiyong-2023.11.7
    Fnet_pretrain_target    = global_vars.Fnet_pretrain_target_normal.squeeze()   # 996 x 1 x 2m_标签无需缩放, yiyong-2023.11.7
    Fnet_pretrain_batchsize = global_vars.Fnet_pretrain_batchsize
    Fnet_pretest_batchsize  = global_vars.Fnet_pretest_batchsize
    epsilon                 = global_vars.Fnet_epsilon
    num_epochs              = global_vars.Fnet_pretrain_num_epoch
    # 模型
    Fnet                    = global_vars.Fnet
    loss                    = torch.nn.MSELoss(reduction='none')
    # pretest里修改了self.hn的batchsize，所以每个epoch要重新初始化hn的的维度
    hn_pretrain             = torch.zeros(Fnet.GRU_num_direction * Fnet.GRU_num_layers, Fnet_pretrain_batchsize, Fnet.GRU_hn_dim).to(dev)
    hn_pretest              = torch.zeros(Fnet.GRU_num_direction * Fnet.GRU_num_layers, Fnet_pretest_batchsize , Fnet.GRU_hn_dim).to(dev)
    #
    LearningRate            = global_vars.Fnet_LearningRate
    Fnet_weightDecay        = global_vars.Fnet_weightDecay
    #
    optimizer               = torch.optim.Adam( Fnet.parameters(), lr= LearningRate,weight_decay=Fnet_weightDecay)

    #########################
    # Pretrain F-net
    #########################
    # global_vars.Fnet.nmse_pretrain_epoch = torch.zeros( [num_epoch] )  # 记录每个epoch更新前的loss
    # global_vars.Fnet.time_pretrain_epoch = torch.zeros( [num_epoch] )  # 记录每个epoch的训练时间  # 100
    # global_vars.Fnet.nmse_pretest_epoch  = torch.zeros([num_epoch])

    global_vars.Fnet.nmse_pretrain_epoch = []  # 记录每个epoch更新前的loss
    global_vars.Fnet.time_pretrain_epoch = []  # 记录每个epoch的训练时间  # 100
    global_vars.Fnet.nmse_pretest_epoch  = []

    epoch = 0
    while True and epoch < num_epochs:

        start_Fnet = time.time()

        Fnet.train()
        h_out = Fnet.Fnet_pretrain_forward( Fnet_pretrain_input, hn_pretrain )   # (996, 2m)

        diff_square1 = loss(h_out, Fnet_pretrain_target.squeeze())  # (996, 2m)
        diff_square2 = torch.sum( diff_square1, dim=1)  # (996,)
        y_square     = torch.norm(Fnet_pretrain_target, p=2, dim=1) ** 2  # 平方和，996
        loss_epoch   = torch.mean( diff_square2 / y_square )  # (1,)

        # 参数更新
        optimizer.zero_grad()
        loss_epoch.backward()
        optimizer.step()

        # 数据存储
        end_Fnet = time.time()
        t_Fnet_epoch = end_Fnet - start_Fnet
        #


        Fnet.nmse_pretrain_epoch.append(loss_epoch)            # 存储本次epoch的nmse
        Fnet.time_pretrain_epoch.append(t_Fnet_epoch)          # 存储本次epoch的训练时间




        torch.cuda.empty_cache()  # 清空缓存

        #########################
        ## 每个Epoch完测试一下性能
        #########################
        optimizer.zero_grad()    # grad =  None
        Fnet_pretest( global_vars, hn_pretest, epoch )
        optimizer.zero_grad()    # grad =  None, 防止梯度计算进入训练流程？有必要吗

        print("Fnet Epoch: {}, Loss:  {:.4f}, time: {:.2f}s, TestLoss:  {:.4f}".format(epoch, Fnet.nmse_pretrain_epoch[epoch],
                                                                    Fnet.time_pretrain_epoch[epoch],Fnet.nmse_pretest_epoch[epoch]))

        global_vars.Fnet_pretrain_num_epoch_actual = epoch
        if epoch > 0 and Fnet.nmse_pretrain_epoch[epoch] < epsilon:
            break  # 如果损失小于epsilon，结束训练

        epoch += 1


#########################
# Fnet pretest函数
#########################
def Fnet_pretest( global_vars, hn_pretest, epoch ):


    dev                      = global_vars.init_dev
    Fnet                     = global_vars.Fnet
    hn_pretest               = hn_pretest
    loss                     = torch.nn.MSELoss(reduction='none')
    # num_epoch                = global_vars.Fnet_pretrain_num_epoch
    mu_y                     = global_vars.Fnet_pretrain_mu_y
    std_y                    = global_vars.Fnet_pretrain_std_y

    Fnet.eval()         # training=False,显式告诉算法进入评估模式，因为dropout和batchnorm在.train()和评估()模式下操作不同
    torch.no_grad()     # 禁用梯度计算(即使requires_grad=True)，禁止那些会被记录在计算图的操作，加快推理速度

    global_vars.nmse_Fnet_pretest = torch.zeros([global_vars.Fnet_pretest_batchsize])

    Fnet_pretest_input                   = global_vars.Fnet_pretest_input_normal.squeeze()    # 100 x p x 2m
    Fnet_pretest_target                  = global_vars.Fnet_pretest_target.squeeze()          # 100 x 2m
                              # num_epoch

    h_out_pretest                        = Fnet.Fnet_pretrain_forward( Fnet_pretest_input, hn_pretest ).unsqueeze(1)   # (100, 2m)
    global_vars.Fnet.h_out_pretest_recovered              = data_recover_Fnet(h_out_pretest, mu_y, std_y)                 # (100, 2m)


    diff_square1                  = loss(global_vars.Fnet.h_out_pretest_recovered.squeeze(), Fnet_pretest_target)   # 平方，100 x 2m
    diff_square2                  = torch.sum(diff_square1, dim=1)                         # 平方和，100
    y_square                      = torch.norm(Fnet_pretest_target, p=2, dim=1) ** 2   # 平方和，100
    global_vars.nmse_Fnet_pretest = diff_square2 / y_square                    # nmse, 100
    global_vars.Fnet.nmse_pretest_epoch.append( torch.mean(global_vars.nmse_Fnet_pretest) ) # nmse，1,



def data_recover_Fnet(data, mu, std):
    x_recover = data * std + mu                   # elemenwise product, yiyong-2023.11.7
    return x_recover







