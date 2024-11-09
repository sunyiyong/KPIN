"""
Main function for KPIN

Author: SUN, Yiyong
Date: 2024.6.30
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
import datetime
import time
import os
import gc

from Global_Variables import Global_Variables
from Complex_AutoRegressive import AutoRegressive
from Complex_StateSpaceModel import StateSpaceModel
from Complex_KFilter import KalmanFilter
from Complex_KPIN_utils import Gnet, split_dataset_Gnet
from Complex_KPIN_pipeline import Gnet_pipeline
from Complex_Plot import Complex_Plot

if torch.cuda.is_available():
    index = 0
    str_dev = "cuda:" + str(index)
    dev = torch.device(str_dev)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Run_main begins on " + str_dev)
else:
    dev = torch.device("cpu")
    print("Run_main begins on CPU")

workingpath = os.getcwd()
seed_values = [123]

# fix random seed
for seed in seed_values:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'

    current_datetime = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    print('Now is ' + current_datetime + '' + '     seed={}'.format(seed))

    # scenario parameters
    vkmh_values = [60]
    f_values = [28]
    N_values = [32]
    M_values = [2]
    SNR_values = [20]
    ms_scale_values = [1]

    file_name_list = []
    folder_index   = []
    for vkmh_idx in range(0, len(vkmh_values)):
        for f_idx in range(0, len(f_values)):
            for N_idx in range(0, len(N_values)):
                for M_idx in range(0, len(M_values)):
                    for SNR_idx in range(0, len(SNR_values)):
                        for ms_idx in range(0, len(ms_scale_values)):
                            f           = f_values[f_idx]
                            N           = N_values[N_idx]
                            vkmh        = vkmh_values[vkmh_idx]
                            M           = M_values[M_idx]
                            SNR         = SNR_values[SNR_idx]
                            ms_scale    = ms_scale_values[ms_idx]

                            # different choices of dynamic condition, k
                            if ms_scale == 1:
                                file_name = r'{}x{}-{}kmh-{}GHz-{}dB'.format(N, M,vkmh, f,SNR)
                            else:
                                file_name = r'{}x{}-{}kmh-{}GHz-{}dB-{}xCoherence'.format(N, M, vkmh, f, SNR, ms_scale)

                            file_name_list.append( file_name )

    for params_idx in range(0, len(file_name_list)):
        # Initialization
        os.chdir(workingpath)
        dataset_file                 = file_name_list[params_idx]
        dataset_folder               = "dataset_channel/3gpp_mmw_UMa/"
        result_folder                = "BatchResult/20240630_KPIN/"
        path                         = dataset_folder + dataset_file + '.mat'
        data                         = loadmat(path)
        global_vars                  = Global_Variables(dev, data)
        global_vars.dataset_folder   = dataset_folder
        global_vars.dataset_file     = dataset_file
        global_vars.result_folder    = result_folder
        global_vars.current_datetime = current_datetime
        print('********************************************')
        print('Dataset: '.format(dataset_folder))
        print('Now params_{} begins: '.format(params_idx) + dataset_file)
        print('SNR: {}dB, num_epoch: {}: '.format(global_vars.init_SNR, global_vars.KPIN_train_num_epoch))
        print('********************************************')

        # AutoRegressive
        global_vars.AR = AutoRegressive()
        global_vars.AR.solve_yule_walker(global_vars)
        global_vars.AR.predict_ar_results(global_vars)
        global_vars.nmse_ar = global_vars.AR.nmse_ar

        # ARKF
        ssm = StateSpaceModel(global_vars)
        global_vars.KFilter = KalmanFilter(ssm)
        global_vars.KFilter.predict_results(global_vars)
        global_vars.nmse_KFilter_hot = global_vars.KFilter.nmse_KFilter_hot

        # KPIN
        global_vars.Gnet = Gnet(global_vars)
        global_vars.Gnet_pipe = Gnet_pipeline(global_vars)
        split_dataset_Gnet(global_vars)
        start = time.time()
        global_vars.Gnet_pipe.KPIN_train(global_vars)
        end = time.time()
        global_vars.time_per_epoch = (end - start) / global_vars.KPIN_train_num_epoch
        print("Time/(epoch) is {:.4f}s".format(global_vars.time_per_epoch))
        torch.cuda.empty_cache()

        # Plot results with Matlab, instead of Pycharm

        # Save all results to specified folder/file
        os.chdir(result_folder)
        torch.save(global_vars, os.path.join('result_pt', dataset_file + '.pt'))


        del global_vars
        gc.collect()
        torch.cuda.empty_cache()




