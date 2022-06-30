import platform, sys, os, pickle, re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import simu.compute_test_statistics as stat
import simu.simulate_data_pd as sim
import simu.simu_mean_detect as mean_detect
import simu.utilities as uti
from sklearn.metrics.cluster import adjusted_rand_score
import pprint
import pandas as pd
#%%
def h_in_IC(changepoints,T):
    '''
    The bonus term for changepoints location in IC
    '''
    return (np.sum(T-1 -changepoints)/np.log(np.sum(T-1 - changepoints)))

def IC(loss, changepoints, g_index, N, T, K, C):
    # """
    # Information criterion
    # Parameters
    # ----------
    # loss : TYPE
    #     DESCRIPTION.
    # C : h(\sum \tau_i) = C \sum \tau_i/log(\sum \tau_i)

    # """
    Kl = K*np.log(np.sum(T-1 -changepoints))
    # print("kl", Kl)
    Ck, indicesList, occurCount = np.unique(g_index, return_index = True,return_counts=True)
    # print("C",C)
    # print('c',occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * C * np.log(np.sum(T-1 -changepoints)))
    # # # print('loss', loss)
    # print('occurCount',occurCount)
    # print('(T-1 -changepoints)[np.s_[indicesList]]',(T-1 -changepoints)[np.s_[indicesList]])
    # print('np.log(np.sum(T-1 -changepoints)',np.log(np.sum(T-1 -changepoints)))
    # print("ic",loss - Kl+ occurCount.dot((T-1 -changepoints)[np.s_[indicesList]]) * C * np.log(np.sum(T-1 -changepoints)))
    return loss - Kl+ occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * C * h_in_IC(changepoints, T)

def paramInIC(model, N, K, T, include_path_loss =0):
    if not include_path_loss:
        Kl_mat =  K*np.log(np.sum(T-1 -model['changepoint']))
        Ck, indicesList, occurCount = np.unique(model['group'], return_index = True,return_counts=True)
        c_mat = occurCount.dot((T-1 -model['changepoint'])[np.s_[indicesList]])/(N*T) * h_in_IC(model['changepoint'], T)
        print("occurCount.dot((T-1 -model['changepoint'])[np.s_[indicesList]])/(N*T)",occurCount.dot((T-1 -model['changepoint'])[np.s_[indicesList]])/(N*T),"h_in_IC(model['changepoint'], T)",h_in_IC(model['changepoint'], T))
        print(c_mat)
        return Kl_mat, c_mat
    else:
        Kl_mat_list = []
        c_mat_list =[]
        cp_eachiter_rev = model['changepoint_eachiter'][:, range(model['changepoint_eachiter'].shape[1]-1,-1,-1)]
        g_eachiter_rev = model['g_index_eachiter'][:, range(model['g_index_eachiter'].shape[1]-1,-1,-1)]
        for ii in range(min(model['changepoint_eachiter'].shape[1], model['g_index_eachiter'].shape[1])):
            Kl_mat_list.append(K*np.log(np.sum(T-1 - cp_eachiter_rev[:,ii])))
            Ck, indicesList, occurCount = np.unique(g_eachiter_rev[:,ii], return_index = True,return_counts=True)
            c_mat_list.append(occurCount.dot((T-1 -cp_eachiter_rev[:,ii])[np.s_[indicesList]])/(N*T) * h_in_IC(cp_eachiter_rev[:,ii], T))
        return Kl_mat_list[::-1], c_mat_list[::-1]
