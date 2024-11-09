"""
This file defines the module we used:

Global_Variables:  for transfering parameters.

Author: SUN, Yiyong
Date: 2024.6.30
"""
import random
import numpy as np
import torch
class Global_Variables:
    def __init__(self, dev, data):
        # load data
        self.init_h                                = data['data']['h'].item()
        self.init_y                                = data['data']['y'].item()
        self.init_y_true                           = data['data']['y_true'].item()
        self.init_noise                            = data['data']['n'].item()
        self.init_rho                              = data['data']['rho'].item().item()
        self.init_M                                = data['data']['M'].item().item()
        self.init_N                                = data['data']['N'].item().item()
        self.init_tau                              = data['data']['tau'].item().item()
        self.init_vkmh                             = data['data']['v_kmh'].item().item()
        self.init_f                                = data['data']['f'].item().item()
        self.init_SNR                              = data['data']['SNR'].item().item()
        self.init_std_vt                           = data['data']['std_vt'].item().item()
        self.init_no_snap                          = data['data']['no_snap'].item().item()
        self.init_l_m                              = data['data']['l_m'].item().item()
        self.init_coherence_time_ms                = data['data']['coherence_time_ms'].item().item()
        self.init_update_rate_ms                   = data['data']['update_rate_ms'].item().item()
        self.ms_scale                              = data['data']['ms_scale'].item().item()
        self.m                                     = self.init_M * self.init_N
        self.n                                     = self.init_tau * self.init_N
        self.p                                     = 4
        self.pm                                    = self.p * self.m
        self.pmn                                   = self.pm * self.n
        self.init_dev                              = dev
        self.Q_pilot_np                            = np.sqrt(self.init_tau) * np.eye(self.init_M, self.init_tau)
        self.Q_np                                  = np.sqrt(self.init_rho) * np.kron(self.Q_pilot_np.T,np.eye(self.init_N))
        self.Q                                     = torch.from_numpy(self.Q_np).to(torch.complex64).to(self.init_dev)
        # AutoRegressive
        self.ar_test_start                         = 2048
        self.ar_test_len                           = 100
        self.ar_test_end                           = self.ar_test_start + self.ar_test_len
        self.ar_train_end                          = self.ar_test_start
        self.ar_train_len                          = 1000
        self.ar_train_start                        = self.ar_train_end - self.ar_train_len
        self.ar_Call_fall_index                    = 0
        self.ar_Call_eps_ratio                     = 1000
        self.ssm_m2x_init_coeff                    = 1e-6
        # ARKF
        self.kfilter_hot_predict_start             = self.ar_train_start
        self.kfilter_hot_predict_len               = self.ar_train_len + self.ar_test_len
        self.kfilter_hot_predict_end               = self.kfilter_hot_predict_start + self.kfilter_hot_predict_len
        self.kfilter_hot_test_start                = self.ar_train_len
        self.kfilter_hot_test_len                  = self.ar_test_len
        self.kfilter_hot_test_end                  = self.kfilter_hot_test_start + self.kfilter_hot_test_len
        self.kfilter_hot_m2y_thres                 = 1e5
        self.kfilter_hot_m2y_fall_index            = 0
        self.kfilter_hot_m2y_eps_ratio             = 1e-10
        # KPIN (Gnet)
        self.Gnet_pretrain_num_epoch               = 100
        self.Gnet_GRU_in_dim                       = 128
        self.Gnet_GRU_hn_dim                       = 256
        self.Gnet_FC2_out_dim                      = 1024
        self.Gnet_supervised_by                    = "y_predict_noisy"
        self.Gnet_CVparams_for_test                = False
        self.Gnet_mismatch_cov_vt                  = 1
        self.Gnet_pretest_start                    = self.ar_test_start
        self.Gnet_pretrain_len                     = self.ar_train_len
        self.Gnet_pretrain_end                     = self.Gnet_pretest_start
        self.Gnet_pretrain_start                   = self.Gnet_pretrain_end - self.Gnet_pretrain_len
        self.Gnet_pretest_len                      = self.ar_test_len
        self.Gnet_pretest_end                      = self.Gnet_pretest_start + self.Gnet_pretest_len
        self.Gnet_pretrain_len_sub                 = 10
        self.Gnet_pretrain_num_all_sub             = round(self.Gnet_pretrain_len / self.Gnet_pretrain_len_sub)
        self.Gnet_pretrain_batchsize               = round(self.Gnet_pretrain_num_all_sub / 2)
        self.Gnet_num_pretest                      = 1
        self.Gnet_pretrain_learningRate            = 5E-5
        self.Gnet_pretrain_weightDecay             = 1E-5
        self.Gnet_pretrain_initSSM_by_KF_posterior = True
        self.noise_ratio                           = 1
        self.h_noise_ratio                         = 5e-8
        self.KPIN_epsilon                          = 0
        self.KPIN_GRUhidden_update                 = 1