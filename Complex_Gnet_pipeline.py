"""
This file defines the module we used:

Gnet_pipeline: to train and test KPIN.

Author: SUN, Yiyong
Date: 2024.6.30
"""

import torch
import torch.nn as nn
import random
import time
from time import strftime, gmtime



class Gnet_pipeline:

    def __init__(self, global_vars ):
        super().__init__()

        self.dev                     = global_vars.init_dev
        self.p                       = global_vars.p
        self.m                       = global_vars.m
        self.n                       = global_vars.n
        self.pm                      = global_vars.pm
        self.Q_re                    = global_vars.Q.real
        self.Q_im                    = global_vars.Q.imag
        #
        self.num_all_sub             = global_vars.Gnet_pretrain_num_all_sub
        self.pretrain_len_sub        = global_vars.Gnet_pretrain_len_sub
        self.num_pretest             = global_vars.Gnet_num_pretest
        self.Gnet_pretest_start      = global_vars.Gnet_pretest_start
        self.pretest_len             = global_vars.Gnet_pretest_len
        self.Gnet_pretrain_len       = global_vars.Gnet_pretrain_len
        #
        self.pretrain_num_epoch      = global_vars.Gnet_pretrain_num_epoch
        self.pretrain_batchsize      = global_vars.Gnet_pretrain_batchsize
        #
        self.pretrain_learningRate   = global_vars.Gnet_pretrain_learningRate
        self.pretrain_weightDecay    = global_vars.Gnet_pretrain_weightDecay
        #
        self.Gnet                    = global_vars.Gnet
        self.optimizer               = torch.optim.Adam(self.Gnet.parameters(), lr=self.pretrain_learningRate, weight_decay=self.pretrain_weightDecay)

    def Gnet_pretrain(self, global_vars):
        self.time_Gnet_pretrain = 0.0
        start_pretrain = time.time()
        pretrain_input = global_vars.Gnet_pretrain_input
        pretrain_target = global_vars.Gnet_pretrain_target
        self.num_epochs = global_vars.Gnet_pretrain_num_epoch
        epsilon = global_vars.KPIN_epsilon

        self.nmse_hLoss_prior_epoch = torch.zeros([self.num_epochs]).to(self.dev)
        self.nmse_hLoss_prior_epoch_dB = torch.zeros([self.num_epochs]).to(self.dev)

        self.nmse_yLoss_epoch = torch.zeros([self.num_epochs]).to(self.dev)
        self.nmse_yLoss_epoch_dB = torch.zeros([self.num_epochs]).to(self.dev)

        self.nmse_hLoss_posterior_epoch = torch.zeros([self.num_epochs]).to(self.dev)
        self.nmse_hLoss_posterior_epoch_dB = torch.zeros([self.num_epochs]).to(self.dev)

        self.nmse_pretest_epoch = torch.zeros([self.num_epochs]).to(self.dev)
        self.nmse_pretest_epoch_dB = torch.zeros([self.num_epochs]).to(self.dev)

        self.Gnet_prior = torch.zeros(self.pretest_len, self.p * self.m,
                                      dtype=torch.complex64).to(self.dev)
        self.Gnet_posterior = torch.zeros(self.pretest_len, self.p * self.m,
                                          dtype=torch.complex64).to(self.dev)

        ti = 0
        # Training
        while True and ti < self.num_epochs:
            self.ti = ti
            self.Gnet.train()

            LOSS_h_prior = torch.tensor([0.]).to(self.dev)
            LOSS_y_prior = torch.tensor([0.]).to(self.dev)
            self.h_prior = torch.zeros(self.pretrain_batchsize, self.m,
                                       self.pretrain_len_sub, dtype=torch.complex64).to(
                self.dev)
            self.x_prior = torch.zeros(self.pretrain_batchsize, self.pm,
                                       self.pretrain_len_sub, dtype=torch.complex64).to(
                self.dev)
            self.y_prior = torch.zeros(self.pretrain_batchsize, self.n,
                                       self.pretrain_len_sub, dtype=torch.complex64).to(
                self.dev)
            self.nmse_hLoss_in_batch = torch.zeros(self.pretrain_batchsize,
                                                   self.pretrain_len_sub).to(self.dev)
            self.nmse_yLoss_in_batch = torch.zeros(self.pretrain_batchsize,
                                                   self.pretrain_len_sub).to(self.dev)

            LOSS_h_posterior = torch.tensor([0.]).to(self.dev)
            self.h_posterior = torch.zeros(self.pretrain_batchsize, self.m,
                                           self.pretrain_len_sub,
                                           dtype=torch.complex64).to(self.dev)
            self.nmse_hLoss_posterior_in_batch = torch.zeros(self.pretrain_batchsize,
                                                             self.pretrain_len_sub).to(
                self.dev)

            start_epoch = time.time()
            for j in range(0, self.pretrain_batchsize):
                if global_vars.Gnet_pretrain_initSSM_by_KF_posterior:
                    n_e = random.randint(1, self.num_all_sub - 1)
                    last_step = n_e * self.pretrain_len_sub - 1
                    self.Gnet.m1x_prior = global_vars.KFilter_hot.KFilter_prior[last_step,
                                          :].to(self.dev)
                    self.Gnet.m1x_posterior = global_vars.KFilter_hot.KFilter_posterior[
                                              last_step, :].to(self.dev)
                else:
                    n_e = random.randint(0, self.num_all_sub - 1)
                    self.Gnet.m1x_prior = torch.zeros(self.pm, dtype=torch.complex64).to(
                        self.dev)
                    self.Gnet.m1x_posterior = torch.zeros(self.pm,
                                                          dtype=torch.complex64).to(
                        self.dev)

                self.Gnet.hn = torch.zeros(1, 1, global_vars.Gnet_GRU_hn_dim).to(self.dev)
                for t in range(0, self.pretrain_len_sub):
                    self.x_prior[j, :, t], self.y_prior[j, :, t] = self.Gnet.one_ssm_loop(
                        pretrain_input[n_e, :, t], global_vars)
                    self.h_prior[j, :, t] = self.x_prior[j, :, t][:self.m]
                    self.h_posterior[j, :, t] = self.Gnet.m1x_posterior[:self.m]
                    torch.cuda.empty_cache()

                    diff_h_abs = torch.abs(
                        self.h_prior[j, :, t] - pretrain_target[n_e, :, t])
                    h_abs = torch.abs(pretrain_target[n_e, :, t])
                    self.nmse_hLoss_in_batch[j, t] = torch.norm(
                        diff_h_abs) ** 2 / torch.norm(h_abs) ** 2

                    diff_y_abs = torch.abs(
                        self.y_prior[j, :, t] - pretrain_input[n_e, :, t])
                    y_abs = torch.abs(pretrain_input[n_e, :, t])
                    self.nmse_yLoss_in_batch[j, t] = torch.norm(
                        diff_y_abs) ** 2 / torch.norm(y_abs) ** 2

                    diff_h_posterior_abs = torch.abs(
                        self.h_posterior[j, :, t] - pretrain_target[n_e, :, t])
                    h_abs = torch.abs(pretrain_target[n_e, :, t])
                    self.nmse_hLoss_posterior_in_batch[j, t] = torch.norm(
                        diff_h_posterior_abs) ** 2 / torch.norm(h_abs) ** 2

                LOSS_h_prior = LOSS_h_prior + torch.mean(self.nmse_hLoss_in_batch, dim=1)[j]
                LOSS_y_prior = LOSS_y_prior + torch.mean(self.nmse_yLoss_in_batch, dim=1)[j]
                LOSS_h_posterior = LOSS_h_posterior + torch.mean(self.nmse_hLoss_posterior_in_batch, dim=1)[j]

            torch.cuda.empty_cache()
            self.nmse_hLoss_in_batch = torch.mean(self.nmse_hLoss_in_batch, dim=1)
            self.nmse_yLoss_in_batch = torch.mean(self.nmse_yLoss_in_batch, dim=1)
            self.nmse_hLoss_posterior_in_batch = torch.mean(self.nmse_hLoss_posterior_in_batch, dim=1)

            self.nmse_hLoss_prior_epoch[ti] = torch.mean(self.nmse_hLoss_in_batch)
            self.nmse_yLoss_epoch[ti] = torch.mean(self.nmse_yLoss_in_batch)
            self.nmse_hLoss_prior_epoch_dB[ti] = 10 * torch.log10(self.nmse_hLoss_prior_epoch[ti])
            self.nmse_yLoss_epoch_dB[ti] = 10 * torch.log10(self.nmse_yLoss_epoch[ti])

            self.nmse_hLoss_posterior_epoch[ti] = torch.mean(
                self.nmse_hLoss_posterior_in_batch)
            self.nmse_hLoss_posterior_epoch_dB[ti] = 10 * torch.log10(
                self.nmse_hLoss_posterior_epoch[ti])

            if global_vars.Gnet_supervised_by == "h_prior":
                LOSS_mean = LOSS_h_prior / self.pretrain_batchsize
            elif global_vars.Gnet_supervised_by == "h_posterior":
                LOSS_mean = LOSS_h_posterior / self.pretrain_batchsize
            elif global_vars.Gnet_supervised_by == "y_predict_noisy":
                LOSS_mean = LOSS_y_prior / self.pretrain_batchsize
            else:
                pass

            self.optimizer.zero_grad()
            LOSS_mean.backward()
            torch.cuda.empty_cache()
            self.optimizer.step()
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()

            end_epoch = time.time()
            self.time_epoch = end_epoch - start_epoch
            self.time_Gnet_pretrain = self.time_Gnet_pretrain + self.time_epoch

            if global_vars.Gnet_supervised_by == "h_prior":
                diff_hLoss = self.nmse_hLoss_prior_epoch_dB[ti] - \
                             self.nmse_hLoss_prior_epoch_dB[ti - 1]
                print(
                    "AR_Gnet Epoch: {},".format(ti),
                    "hLoss: {:.2f}[dB],".format(
                        self.nmse_hLoss_prior_epoch_dB[ti].item()),
                    "diff_hLoss: {:.2f}[dB]".format(diff_hLoss),
                    "time: {:.2f}[s]".format(self.time_epoch)
                )
            elif global_vars.Gnet_supervised_by == "h_posterior":
                diff_hLoss_posterior = self.nmse_hLoss_posterior_epoch_dB[ti] - \
                                       self.nmse_hLoss_posterior_epoch_dB[ti - 1]
                print(
                    "AR_Gnet Epoch: {},".format(ti),
                    "hLoss: {:.2f}[dB],".format(
                        self.nmse_hLoss_posterior_epoch_dB[ti].item()),
                    "diff_hLoss: {:.2f}[dB]".format(diff_hLoss_posterior),
                    "time: {:.2f}[s]".format(self.time_epoch)
                )
            elif global_vars.Gnet_supervised_by == "y_predict_noisy":
                diff_yLoss = self.nmse_yLoss_epoch_dB[ti] - self.nmse_yLoss_epoch_dB[
                    ti - 1]
                print(
                    "AR_Gnet Epoch: {},".format(ti),
                    "yLoss: {:.2f}[dB],".format(self.nmse_yLoss_epoch_dB[ti].item()),
                    "diff_yLoss: {:.2f}[dB]".format(diff_yLoss),
                    "time: {:.2f}[s]".format(self.time_epoch)
                )

            if self.ti == self.num_epochs - 1:
                self.Gnet_pretest(global_vars)
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

            if self.ti == self.num_epochs - 1:
                end_pretrain = time.time()
                time_pretrain = end_pretrain - start_pretrain
                print(
                    "Total Train Time: " + strftime("%H:%M:%S", gmtime(time_pretrain))
                )

            if ti > 0 and LOSS_mean < epsilon:
                self.Gnet_pretest(global_vars)
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                break

            ti += 1
            torch.cuda.empty_cache()

        global_vars.Gnet_pretrain_num_epoch_actual = ti

    def Gnet_pretest(self, global_vars):
        test_input = global_vars.Gnet_pretest_input
        test_target = global_vars.Gnet_pretest_target

        self.Gnet.eval()
        torch.no_grad()

        self.out_test_h = torch.zeros(self.num_pretest, self.m, self.pretest_len,
                                      dtype=torch.complex64).to(self.dev)
        self.out_test_x = torch.zeros(self.num_pretest, self.pm, self.pretest_len,
                                      dtype=torch.complex64).to(self.dev)
        self.out_test_y = torch.zeros(self.num_pretest, self.n, self.pretest_len,
                                      dtype=torch.complex64).to(self.dev)
        self.nmse_Gnet_pretest = torch.zeros(self.num_pretest, self.pretest_len).to(
            self.dev)
        self.out_test_KG = torch.zeros(self.num_pretest, self.pretest_len, self.pm,
                                       self.n, dtype=torch.complex64).to(self.dev)

        start_test = time.time()
        for j in range(0, self.num_pretest):
            self.Gnet.m1x_prior = global_vars.KFilter_hot.KFilter_prior[
                                  self.Gnet_pretrain_len - 1, :].to(self.dev)
            self.Gnet.m1x_posterior = global_vars.KFilter_hot.KFilter_posterior[
                                      self.Gnet_pretrain_len - 1, :].to(self.dev)
            self.Gnet.hn = torch.zeros(1, 1, global_vars.Gnet_GRU_hn_dim).to(self.dev)

            for t in range(0, self.pretest_len):
                self.out_test_x[j, :, t], self.out_test_y[j, :,
                                          t] = self.Gnet.one_ssm_loop(test_input[j, :, t],
                                                                      global_vars)
                self.out_test_h[j, :, t] = self.out_test_x[j, :, t][:self.m]

                if self.ti == global_vars.Gnet_pretrain_num_epoch - 1:
                    self.Gnet_prior[t, :] = self.Gnet.m1x_prior
                    self.Gnet_posterior[t, :] = self.Gnet.m1x_posterior

                diff_h_abs = torch.abs(self.out_test_h[j, :, t] - test_target[j, :, t])
                h_abs = torch.abs(test_target[j, :, t])
                self.nmse_Gnet_pretest[j, t] = torch.norm(diff_h_abs) ** 2 / torch.norm(
                    h_abs) ** 2
                self.out_test_KG[j, t, :, :] = torch.complex(self.Gnet.KG_re,
                                                             self.Gnet.KG_im)

        end_test = time.time()
        self.time_test = end_test - start_test

        self.nmse_pretest_epoch[self.ti] = torch.mean(self.nmse_Gnet_pretest, dim=1)
        self.nmse_pretest_epoch_dB[self.ti] = 10 * torch.log10(
            self.nmse_pretest_epoch[self.ti])
        diff_yLoss = self.nmse_pretest_epoch_dB[self.ti] - self.nmse_pretest_epoch_dB[
            self.ti - 1]
        print("              nmse_test: {:.2f}[dB]".format(
            self.nmse_pretest_epoch_dB[self.ti]),
              "  diff_yLoss: {:.2f}[dB]".format(diff_yLoss),
              " time: {:.2f}[s]".format(self.time_test))

        torch.cuda.empty_cache()






