"""
This file defines the module we used:

StateSpaceModel: which is for implementing KPIN and ARKF.

Author: SUN, Yiyong
Date: 2024.6.30
"""
import torch
import numpy as np


class StateSpaceModel:

    def __init__(self, global_vars):
        self.dev               = global_vars.init_dev
        Phi                    = global_vars.AR.Phi
        cov_ut                 = global_vars.AR.cov_ut
        P                      = global_vars.AR.P
        cov_vt                 = global_vars.AR.cov_vt
        kfilter_m2x_init_coeff = global_vars.ssm_m2x_init_coeff
        tau                    = global_vars.init_tau
        p                      = global_vars.p
        M                      = global_vars.init_M
        N                      = global_vars.init_N
        m                      = M*N
        n                      = tau*N

        self.Phi               = torch.from_numpy(Phi).to(torch.complex64).to(self.dev)
        self.P                 = torch.from_numpy(P).to(torch.complex64).to(self.dev)
        self.M                 = M
        self.N                 = N
        self.p                 = p
        self.tau               = tau
        self.rho               = global_vars.init_rho
        self.m                 = m
        self.n                 = n

        #Transition Model
        eye_matrix        = np.eye( (p-1)*m )
        zeros_matrix      = np.zeros( ((p-1)*m, m) )
        F                 = np.hstack( [eye_matrix, zeros_matrix] )
        F                 = np.vstack( [Phi, F] )
        self.F            = torch.from_numpy(F).to(torch.complex64).to(self.dev)
        self.F_H          = torch.conj( torch.transpose(self.F, 0, 1) )

        self.cov_ut       = torch.from_numpy(cov_ut).to(torch.complex64).to(self.dev)
        theta             = np.vstack( [np.eye(m), zeros_matrix] )
        self.theta        = torch.from_numpy(theta).to(torch.complex64).to(self.dev)
        self.theta_H      = torch.conj( torch.transpose(self.theta, 0, 1) )

        #Measurement Model
        zero_vector = np.zeros( (tau*N, (p-1)*m) )
        G           = np.hstack( [P, zero_vector] )
        self.G      = torch.from_numpy(G).to(torch.complex64).to(self.dev)
        self.G_H    = torch.conj( torch.transpose( self.G, 0, 1) )

        self.cov_vt = torch.from_numpy(cov_vt).to(torch.complex64).to(self.dev)

        #Initial latent state
        self.m1x_0  = torch.squeeze( torch.zeros((p*m, 1), dtype = torch.complex64) ).to(self.dev)  # first-order moment
        self.m2x_0  = kfilter_m2x_init_coeff * torch.eye( p*m, dtype= torch.complex64).to(self.dev)  # second-order moment


