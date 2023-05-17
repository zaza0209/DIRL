import pickle
from glob import glob
import platform
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re, subprocess
plat = platform.platform()
if plat == 'macOS-13.0-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private/simulation_real")
    sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")
elif plat == 'Linux-4.18.0-305.65.1.el8_4.x86_64-x86_64-with-glibc2.28':  # greatlakes
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")
from functions.utilities import *



#%%
effect_sizes = ["weak", "moderate","strong"] #
# effect_sizes = ["strong"] #
Ks = [2,3,4] 
inits = [15,20]
# date = '20221031_C0'
date = '20230510'
random_seeds = np.arange(1, 20+1, step = 1)

# def h_in_IC(changepoints,T, h = '1'):
#     '''
#     The bonus term for changepoints location in IC
#     '''
#     if h == '1':
#         return (np.sum(T-1 -changepoints)/np.log(np.sum(T-1 - changepoints)))
#     elif h == 'log':
#         return (np.log(np.sum(T - 1 - changepoints)))
#     elif h == 'sqrt':
#         return (np.sqrt(np.sum(T-1 -changepoints)/np.log(np.sum(T-1 - changepoints))))
#
# def IC(loss, changepoints, g_index, N, T, K, C=0, Kl_fun='log', h='1'):
#     if Kl_fun == 'log':
#         Kl = K*np.log(np.sum(T-1 -changepoints))
#     elif Kl_fun == "sqrt":
#         Kl = K*np.sqrt(np.sum(T-1 -changepoints))
#     # print("kl", Kl)
#     Ck, indicesList, occurCount = np.unique(g_index, return_index = True,return_counts=True)
#     loss_ = loss / np.mean(T-1-changepoints)
#     third_term = occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * C * h_in_IC(changepoints, T, h=h)
#     # print("ic =", loss_ - Kl+ occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * C * h_in_IC(changepoints, T), ',',
#     #       K, '*l =', Kl,  ', ', C,'* h =', C*occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * h_in_IC(changepoints, T))
#     return loss_ - Kl / 2*T + third_term / 2*T


C = 1
N = 150
T = 26
h = '1'
dat = pd.DataFrame(columns=["Effect Size", 'seed', 'K', 'init', 'IC', 'ARI', 'cp_error', 'iterations'])
# print(dat)
row_idx = 0
for effect_size in effect_sizes:
    print(effect_size)
    path_name = 'data/' + date + '/sim_' + effect_size + "/"
    for K in Ks:
        # print(K)
        for init in inits:
            print(init)
            for seed in random_seeds:
                file_name = path_name + "result_seed" + str(seed) + "_K" + str(K) + "_init" + str(init) + ".dat"
                try:
                    out = pickle.load(open(file_name, "rb"))
                    loss = out['loss']
                    changepoints = out['changepoints']
                    g_index = out['clusters']
                    value = IC(loss, changepoints, g_index, N, T, K, C, Kl_fun='Nlog(NT)/T', h=h)

                    dat.loc[row_idx] = [effect_size, seed, out['K'], init, value, out["cluster_ari"], out["changepoint_err"], out["iter_num"]]
                    row_idx += 1
                except:
                    print("Error:", file_name)
                    continue


print(dat)
dat.to_csv("output/" + date + "_offline_raw.csv")
# file_name = "output/" + date + "_offline.csv"
# with open(file_name, 'wb') as f:
#     pickle.dump(dat, f)


# dat_summarized = dat.groupby(["Effect Size", 'seed'], as_index=False).agg({'IC': 'max'}).reset_index()
dat_summarized = dat.sort_values("IC", ascending=False).groupby(["Effect Size", 'seed'], as_index=False).first()
print(dat_summarized)
dat_summarized.to_csv("output/" + date + "_offline_summarized.csv")

# subprocess.call(["module load R"])
subprocess.call(["Rscript", "--vanilla", "p01_plot_sim_result.R", "output/" + date + "_offline_summarized.csv"])