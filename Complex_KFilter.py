"""
This file defines the module we used:

KalmanFilter_hot: which is to implement ARKF.

Author: SUN, Yiyong
Date: 2024.6.30
"""
import torch
import numpy as np



class KalmanFilter_hot:
    # Initialization
    def __init__(self, StateSpaceModel):

        self.dev           = StateSpaceModel.dev

        self.F             = StateSpaceModel.F
        self.F_H           = StateSpaceModel.F_H
        self.m             = StateSpaceModel.m

        self.cov_ut        = StateSpaceModel.cov_ut

        self.theta         = StateSpaceModel.theta
        self.theta_H       = StateSpaceModel.theta_H

        self.G             = StateSpaceModel.G
        self.G_H           = StateSpaceModel.G_H
        self.n             = StateSpaceModel.n

        self.cov_vt        = StateSpaceModel.cov_vt

        self.m1x_0         = StateSpaceModel.m1x_0  # first-order moment
        self.m2x_0         = StateSpaceModel.m2x_0  # second-order moment

        self.M             = StateSpaceModel.M
        self.N             = StateSpaceModel.N
        self.p             = StateSpaceModel.p
        self.tau           = StateSpaceModel.tau


    # predict results for the total trajectory
    def predict_results(self, global_vars):
        y                   = torch.from_numpy(global_vars.init_y).to(torch.complex64).to(self.dev)
        h                   = torch.from_numpy(global_vars.init_h).to(torch.complex64).to(self.dev)
        hot_predict_start   = global_vars.kfilter_hot_predict_start
        hot_predict_len     = global_vars.kfilter_hot_predict_len
        hot_predict_end     = global_vars.kfilter_hot_predict_end

        self.m2y_eps_ratio  = global_vars.kfilter_hot_m2y_eps_ratio
        self.m2y_fall_index = global_vars.kfilter_hot_m2y_fall_index
        self.m2y_thres      = global_vars.kfilter_hot_m2y_thres

        index_save          = global_vars.kfilter_hot_test_start

        y_test_kfilter_hot      = y[:, hot_predict_start: hot_predict_end]
        self.h_test_kfilter_hot = h[:, hot_predict_start: hot_predict_end]
        self.nmse_KFilter_hot   = torch.zeros(hot_predict_len)
        self.KFilter_prior      = torch.zeros(hot_predict_len, self.p * self.m,dtype=torch.complex64)
        self.KFilter_posterior  = torch.zeros(hot_predict_len, self.p * self.m,dtype=torch.complex64)
        self.InitSequence()
        for t in range(0, hot_predict_len):
            self.KF_one_loop(y_test_kfilter_hot[:, t])
            self.KFilter_prior[t, :]     = self.m1x_prior
            self.KFilter_posterior[t, :] = self.m1x_posterior

            if t == index_save - 1:
                global_vars.kformer_m1x_0 = self.m1x_posterior

            diff_abs = torch.abs(self.m1x_prior[:self.m] - self.h_test_kfilter_hot[:, t])
            h_abs    = torch.abs(self.h_test_kfilter_hot[:, t])
            self.nmse_KFilter_hot[t] = torch.norm(diff_abs) ** 2 / torch.norm(h_abs) ** 2

    def InitSequence(self):
        self.m1x_posterior = self.m1x_0  # first-order moment
        self.m2x_posterior = self.m2x_0  # second-order moment

    def KF_one_loop(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.matmul( self.F, self.m1x_posterior)

        # Predict the 2-nd moment of x
        a = torch.matmul( self.F, self.m2x_posterior )
        a = torch.matmul( a, self.F_H )
        b = torch.matmul( self.theta, self.cov_ut )
        b = torch.matmul( b, self.theta_H )
        self.m2x_prior = a + b

        # Predict the 1-st moment of y
        self.m1y = torch.matmul( self.G, self.m1x_prior )

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul( self.G, self.m2x_prior)
        self.m2y = torch.matmul( self.m2y, self.G_H) + self.cov_vt


    # Compute the Kalman Gain
    def KGain(self):
        self.KG                 = torch.matmul( self.m2x_prior, self.G_H )
        m2y_cpu                 = self.m2y.cpu().detach().numpy()
        self.m2y_inv     = np.linalg.inv( m2y_cpu )
        self.m2y_inv_gpu = torch.from_numpy(self.m2y_inv).to(torch.complex64).to(self.dev)
        self.KG          = torch.matmul( self.KG, self.m2y_inv_gpu )

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul( self.KG, self.dy )

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul( self.m2y, torch.conj(torch.transpose(self.KG,0,1)) )
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)


