"""
This file defines the module we used:

AutoRegressive class: which is to compute the AR parameters for SSM construction

Author: SUN, Yiyong
Date: 2024.6.30
"""
import torch
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import datetime


class AutoRegressive:

   def __init__(self):
      pass

   def solve_yule_walker(self, global_vars):
      rho = global_vars.init_rho
      M = global_vars.init_M
      N = global_vars.init_N
      p = global_vars.p
      tau = global_vars.init_tau
      ar_train_start = global_vars.ar_train_start
      ar_train_len = global_vars.ar_train_len
      ar_eps_ratio = global_vars.ar_Call_eps_ratio
      std_vt = global_vars.init_std_vt

      print("Yule-Walker Equations for AR(" + str(p) + ") begins")

      y_train_temp = global_vars.init_y[:, ar_train_start: ar_train_start + ar_train_len]
      mean_y = np.mean(y_train_temp, 1)
      y_train = y_train_temp - mean_y[:, np.newaxis]
      y_train = y_train_temp

      P = np.sqrt(tau) * np.eye(M, tau)
      P = np.sqrt(rho) * np.kron(P.T, np.eye(N))
      self.cov_vt = (std_vt ** 2) * np.eye(tau * N)

      # Yule-Walker Equations
      for k in range(0, p + 1):
         if k == 0:
            Y_tk    = y_train
            Y_t     = y_train
            Ck_y    = 1 / (ar_train_len - 1) * np.dot( Y_tk, np.conj(Y_t).T )
            Ck_temp = np.dot( np.linalg.pinv(P), Ck_y-self.cov_vt )
            Ck      = np.dot( Ck_temp, np.linalg.pinv(np.conj(P).T) )
            Ctemp   = Ck
         else:
            Y_tk     = y_train[:, :-k]
            Y_t      = y_train[:, k:]
            Ck_y     = 1 / (ar_train_len - 1) * np.dot( Y_tk, np.conj(Y_t).T )
            Ck_temp  = np.dot( np.linalg.pinv(P), Ck_y )
            Ck       = np.dot( Ck_temp, np.linalg.pinv( np.conj(P).T ))
            Ctemp    = np.vstack( [Ctemp, Ck] )

      # The matrix C_all
      self.C_all = np.zeros( (p*M*N, p*M*N), dtype=np.complex64 )
      for i in range(0, p):
         for j in range(0, p):
            if i >= j:
               self.C_all[i*M*N:(i+1)*M*N, j*M*N:(j+1)*M*N] = Ctemp[(i-j)*M*N:(i-j+1)*M*N,:]
            else:
               self.C_all[i*M*N:(i+1)*M*N, j*M*N:(j+1)*M*N] = np.conj( Ctemp[(j-i)*M*N:(j-i+1)*M*N,:] ).T

      # # Small Perturbation
      self.ar_eig_vals       = np.linalg.eigvals( self.C_all )
      self.ar_eig_abs        = np.abs( self.ar_eig_vals )
      self.ar_sorted_indices = np.argsort(self.ar_eig_abs)[::-1]
      self.ar_eig_abs_sorted = self.ar_eig_abs[self.ar_sorted_indices]
      self.ar_eps            = self.ar_eig_abs_sorted[global_vars.ar_Call_fall_index] / ar_eps_ratio
      C_all_perturbed        = self.C_all + self.ar_eps * np.eye(p*M*N)

      # solve Y-W Equations
      Phi_H = np.linalg.solve( C_all_perturbed, Ctemp[M*N:,:] )
      self.Phi = np.conj( Phi_H ).T
      self.cov_ut_wrong = Ctemp[:M*N,:] - np.dot( self.Phi, Ctemp[M*N:,:] )
      self.cov_ut = Ctemp[:M*N,:] - np.dot( Ctemp[M*N:,:].T.conj(),self.Phi.T.conj() )

      self.P = P
      self.ar_params = {
         'C_all': self.C_all,
         'ar_eig_abs_sorted': self.ar_eig_abs_sorted,
         'Phi': self.Phi,
         'cov_ut': self.cov_ut,
         'P': self.P,
         'cov_vt': self.cov_vt
      }

      return self.ar_params

   # Prediction using AutoRegressive
   def predict_ar_results(self, global_vars):
      M = global_vars.init_M
      N = global_vars.init_N
      p = global_vars.p

      ar_test_start = global_vars.ar_test_start
      ar_test_len = global_vars.ar_test_len

      self.h_pre = np.zeros([M * N, ar_test_len], dtype=np.complex64)
      self.nmse_ar = np.zeros(ar_test_len)

      sample_ut = self.generate_complex_sample_EVD(global_vars, self.cov_ut)
      for t in range(0, ar_test_len):
         h_past = global_vars.init_h[:, ar_test_start + t - p:ar_test_start + t]
         h_past_reversed = h_past[:, ::-1]
         h_past_vstack = h_past_reversed.T.flatten().T
         self.h_pre[:, t] = np.dot(self.Phi, h_past_vstack) + sample_ut[:, t]

         h_true = global_vars.init_h[:, ar_test_start + t]
         h_true_abs = np.abs(h_true)
         diff_abs = np.abs(self.h_pre[:, t] - h_true)
         self.nmse_ar[t] = np.linalg.norm(diff_abs) ** 2 / np.linalg.norm(h_true_abs) ** 2

      return self.nmse_ar

   # sample from ut
   def generate_complex_sample_EVD(self, global_vars, Matrix):
      assert np.allclose(Matrix,Matrix.T.conj()), "$init_M$不是 Hermitian 矩阵，不能执行Chelosky分解"

      len = global_vars.ar_test_len
      m = Matrix.shape[0]
      mean = np.zeros(m)
      covariance = np.eye(m)
      a = np.random.multivariate_normal(mean, covariance, len).T

      eigenvalues, eigenvectors = np.linalg.eig(Matrix)
      A_sqrt = np.diag(np.sqrt(eigenvalues))
      L = eigenvectors.dot(A_sqrt)
      sample = np.dot(L, a)

      return sample










































