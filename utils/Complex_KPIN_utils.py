"""
This file defines the module we used:

Gnet Class: our proposed KPIN module.

Author: SUN, Yiyong
Date: 2024.6.30
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func



class Gnet(torch.nn.Module):
    def __init__(self, global_vars):
        super().__init__()

        self.dev     = global_vars.init_dev
        self.p       = global_vars.p
        self.m       = global_vars.m
        self.n       = global_vars.n
        self.pm      = global_vars.pm
        self.pmn     = global_vars.pmn
        self.KPIN_GRUhidden_update = global_vars.KPIN_GRUhidden_update

        Phi          = global_vars.AR.Phi
        eye_matrix   = np.eye((self.p - 1) * self.m)
        zeros_matrix = np.zeros(((self.p - 1) * self.m, self.m))
        F            = np.hstack([eye_matrix, zeros_matrix])
        F            = np.vstack([Phi, F])
        self.F       = torch.from_numpy(F).to(torch.complex64).to(self.dev)
        self.F_re    = self.F.real
        self.F_im    = self.F.imag


        self.FC1_in_dim        = 2 * ( self.p * self.m + self.n )
        self.GRU_in_dim        = global_vars.KPIN_GRU_in_dim
        self.GRU_hn_dim        = global_vars.KPIN_GRU_hn_dim
        self.GRU_num_layers    = 1
        self.GRU_num_direction = 1
        self.GRU_batch_size    = 1
        self.GRU_seq_len       = 1
        self.FC2_out_dim       = global_vars.KPIN_FC2_out_dim
        self.FC3_out_dim       = 2 * self.p * self.m * self.n

        #Gnet construction
        # FC1
        self.l1_linear         = nn.Linear(self.FC1_in_dim, self.GRU_in_dim, bias=True).to(self.dev)
        self.l1_relu           = nn.ReLU()

        # GRU
        self.Gnet_GRU          = nn.GRU(self.GRU_in_dim, self.GRU_hn_dim, self.GRU_num_layers, batch_first=True).to(self.dev)
        self.hn                = torch.zeros(self.GRU_num_direction * self.GRU_num_layers, self.GRU_batch_size,self.GRU_hn_dim).to(self.dev)


        # FC2
        self.l2_linear         = nn.Linear(self.GRU_hn_dim, self.FC2_out_dim, bias=True).to(self.dev)  # hn_dim x FC2_out_dim
        self.l2_relu           = nn.ReLU()

        # FC3, no activation
        self.l3_linear         = nn.Linear(self.FC2_out_dim, self.FC3_out_dim, bias=True).to(self.dev)  # FC2_out_dim x 2m



    def one_ssm_loop(self, yt, global_vars):
        self.m1x_prev_prior = self.m1x_prior
        m1x_posterior_re = self.m1x_posterior.real
        m1x_posterior_im = self.m1x_posterior.imag

        # compute prior
        m1x_prior_re = torch.matmul(self.F_re, m1x_posterior_re) - torch.matmul(self.F_im, m1x_posterior_im)
        m1x_prior_im = torch.matmul(self.F_re, m1x_posterior_im) + torch.matmul(self.F_im, m1x_posterior_re)
        self.m1x_prior = torch.complex(m1x_prior_re, m1x_prior_im)

        # expected observation
        h_prior_re = m1x_prior_re[:self.m]
        h_prior_im = m1x_prior_im[:self.m]
        m1y_re = torch.matmul(global_vars.Q.real, h_prior_re) - torch.matmul(global_vars.Q.imag, h_prior_im)
        m1y_im = torch.matmul(global_vars.Q.real, h_prior_im) + torch.matmul(global_vars.Q.imag, h_prior_re)
        self.m1y = torch.complex(m1y_re, m1y_im)

        # data-driven weighting matrix
        self.compute_KGain(yt)

        dy    = yt - self.m1y
        dy_re = dy.real
        dy_im = dy.imag

        # Compute the posterior
        INOV_re = torch.matmul(self.KG_re, dy_re) - torch.matmul(self.KG_im, dy_im)
        INOV_im = torch.matmul(self.KG_re, dy_im) + torch.matmul(self.KG_im, dy_re)
        INOV    = torch.complex(INOV_re, INOV_im)
        self.m1x_posterior = self.m1x_prior + INOV

        return [self.m1x_prior, self.m1y]




    # data-driven weighting matrix
    def compute_KGain(self, y):
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_temp = torch.cat([dm1x.real, dm1x.imag])
        dm1x_temp = torch.squeeze(dm1x_temp)
        dm1x_norm = func.normalize(dm1x_temp, p=2, dim=0, eps=1e-12, out=None)

        dm1y = y - self.m1y
        dm1y_temp = torch.cat([dm1y.real, dm1y.imag])
        dm1y_temp = torch.squeeze(dm1y_temp)
        dm1y_norm = func.normalize(dm1y_temp, p=2, dim=0, eps=1e-12, out=None)

        GNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)

        self.KG_neuron = self.Gnet_pretrain_forward(GNet_in).squeeze()

        self.KG_re = self.KG_neuron[:self.pmn].reshape(self.pm, self.n)
        self.KG_im = self.KG_neuron[self.pmn:].reshape(self.pm, self.n)

    # forward of KPIN
    def Gnet_pretrain_forward(self, input):
        l1_out = self.l1_linear(input)
        la1_out = self.l1_relu(l1_out)

        GRU_in = la1_out.unsqueeze(0).unsqueeze(0)
        if self.KPIN_GRUhidden_update:
            GRU_out, self.hn = self.Gnet_GRU(GRU_in, self.hn)
        else:
            GRU_out, a = self.Gnet_GRU(GRU_in, self.hn)

        GRU_out_reshape = GRU_out.squeeze()

        l2_out = self.l2_linear(GRU_out_reshape)
        la2_out = self.l2_relu(l2_out)

        l3_out = self.l3_linear(la2_out)

        return l3_out


# Data Preprocessing
def split_dataset_Gnet( global_vars ):

    dev          = global_vars.init_dev
    y_true       = torch.from_numpy(global_vars.init_y_true).to(torch.complex64).to(dev)
    noise        = torch.from_numpy(global_vars.init_noise).to(torch.complex64).to(dev)
    y            = y_true + global_vars.noise_ratio * noise
    h            = torch.from_numpy(global_vars.init_h).to(torch.complex64).to(dev)

    train_start  = global_vars.KPIN_train_start
    test_start   = global_vars.KPIN_test_start
    num_pretrain = global_vars.KPIN_train_num_all_sub
    len_pretrain = global_vars.KPIN_train_len_sub

    num_pretest  = global_vars.KPIN_num_test
    len_pretest  = global_vars.KPIN_test_len
    input_dim    = y.shape[0]
    output_dim   = h.shape[0]

    assert input_dim == global_vars.init_tau * global_vars.init_N, "KPIN_GRU_in_dim 「init_y」 does not match the expected value."
    assert output_dim == global_vars.init_M * global_vars.init_N, "output_dim does 「init_h」 not match the expected value."
    assert num_pretrain * len_pretrain + num_pretest * len_pretest <= y.shape[1], "Total sequence length exceeds input sequence length."


    train_input = torch.zeros(num_pretrain, input_dim, len_pretrain, dtype=torch.complex64, device=dev)
    train_target = torch.zeros(num_pretrain, output_dim, len_pretrain, dtype=torch.complex64, device=dev)

    start_index = train_start
    for i in range(num_pretrain):
        end_index = start_index + len_pretrain
        train_input[i] = y[:, start_index:end_index]
        train_target[i] = h[:, start_index:end_index]
        start_index = end_index

    test_input = torch.zeros(num_pretest, input_dim, len_pretest, dtype=torch.complex64, device=dev)
    test_target = torch.zeros(num_pretest, output_dim, len_pretest, dtype=torch.complex64, device=dev)

    start_index = test_start
    for i in range(num_pretest):
        end_index = start_index + len_pretest
        test_input[i] = y[:, start_index:end_index]
        test_target[i] = h[:, start_index:end_index]
        start_index = end_index

    global_vars.Gnet_pretrain_input  = train_input
    global_vars.Gnet_pretrain_target = train_target
    global_vars.Gnet_pretest_input   = test_input
    global_vars.Gnet_pretest_target  = test_target