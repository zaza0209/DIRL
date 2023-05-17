import platform
import sys
import os, subprocess
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

plat = platform.platform()
print(plat)
if plat == 'macOS-13.0-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private/simulation_real")
    sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")
elif plat == 'Linux-4.18.0-305.65.1.el8_4.x86_64-x86_64-with-glibc2.28':  # greatlakes
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")

import re

effect_sizes = ["strong", "moderate", "weak"] #"veryverystrong", "verystrong",
date = '20230510'
nsim = 20

#%%

def replace_na(x):
    if x == -999:
        return np.nan
    else:
        return x

def load_replace_na(file_name):
    try:
        out = replace_na(pickle.load(open(file_name, "rb")))
        return out
    except:
        print("cannot open", file_name)
        out = {}
        out['value'] = {}
        out['value']['average_reward'] = -999
        out['value']['discounted_reward'] = -999
        return out


type_ests = [
    'proposed',
    'overall',
    'oracle',
    'cluster_only',
    'changepoint_only'
]
file_name_labels = ['Proposed', #'Proposed (agnostic 0)', 'Proposed (agnostic 1)',
                    'Overall', #'Overall (agnostic 0)', 'Overall (agnostic 1)',
                    'Oracle', 'Random'] #

df = pd.DataFrame(columns=['Effect Size', 'seed',
                           'Method', 'Discounted Reward', 'Average Reward'])
row_idx_table = 0
row_idx = 0


for effect_size in effect_sizes:
    print(effect_size)
    data_path0 = 'data/' + date + '/' + 'value_'  + effect_size + "/"
    for nrep in range(1, nsim + 1):
        for type_est in type_ests:
            file_name = data_path0 + "seed" + str(nrep) + "/value_online_" + type_est + ".dat"
            # try:
            values = load_replace_na(file_name)
            # raw_reward = values['raw_reward']
            average_reward = values['value']['average_reward']
            discounted_reward = values['value']['discounted_reward']
            # if nrep == 1:
            #     print(raw_reward.shape)
            # discounted reward
            # discounted_reward = 0.0
            # T_initial = 100
            # for t in range(T_initial, raw_reward.shape[1]):
            #     discounted_reward += raw_reward[:, t] * gamma ** (t - T_initial)
            # discounted_reward = np.mean(discounted_reward)
            # average_reward = np.mean(raw_reward[:, 100:])
            # values['values']['discounted_reward'] = discounted_reward
            # values['values']['average_reward'] = average_reward
            # with open(file_name, "wb") as f:
            #     pickle.dump(values, f)
            df.loc[row_idx] = [effect_size, nrep, type_est, discounted_reward, average_reward]
            row_idx += 1
            # except:
            #     print(file_name + " does not exist.")
print(df)

df.to_csv('output/' + date + '_optvalue_online.csv', index=False)




