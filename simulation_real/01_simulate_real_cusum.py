'''
Simulate nonstationary time series data and apply Q-learning.
Data generation mechanism resembles the IHS data
'''
#!/usr/bin/python
import platform, sys, os, pickle, re
plat = platform.platform()
# print(plat)
if plat == 'macOS-12.4-x86_64-i386-64bit' or plat == 'macOS-10.16-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL/simulation_real")
    sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")
elif plat == 'Linux-3.10.0-1160.6.1.el7.x86_64-x86_64-with-glibc2.17' or plat == 'Linux-3.10.0-1160.45.1.el7.x86_64-x86_64-with-glibc2.17':  # greatlakes
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import simu.simu_mean_detect as mean_detect
from sklearn.metrics.cluster import adjusted_rand_score
import simulation_real.simulate_data_real as sim
import copy
import pandas as pd
from simu.utilities import *
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from functions.evaluation_separateA import *
# import HeterRL.functions.compute_test_statistics_separateA as stat


# Arguments passed

# # argv should be: seed, kappa, degree, num_threads
seed = int(sys.argv[1])
effect_size = str(sys.argv[2])
threshold_type = str(sys.argv[3]) #"Chi2" or "permutation"
# kappa = int(sys.argv[2])
# gamma = float(sys.argv[3])
# N = int(sys.argv[4])
# seed = 90
# kappa = 10
# gamma = 0.9

# import importlib
# importlib.reload(sim)
# importlib.reload(mean_detect)

startTime = datetime.now()
np.random.seed(seed)
print("seed =", seed)

# %% simulate data
# terminal timestamp
T = 26
#change point
# change_pt = 16
# number of people
# N = int(3)
num_threads = 5
test_setting = "cdist"
# effect_size = "large"
if effect_size == "small":
    effect_size_factor = 0.3
elif effect_size == "moderate":
    effect_size_factor = 0.7
elif effect_size == "large":
    effect_size_factor = 1.0
cluster_settings = {'cluster1': {'n': 25, 'changepoints': [11],
                                 'state': [np.array([[10, 0.4, -0.04, 0.1],
                                                    [11, -0.4, 0.05, 0.4],
                                                    [1.2, -0.02, 0.03, 0.8]]),
                                           np.array([[11, 0.4, -0.04, 0.1],
                                                     [12, -0.4, 0.05, 0.4],
                                                     [1.6, -0.02, 0.03, 0.8]])
                                           ],
                                 'action': [np.array([[0.6, 0.3, 0, 0],
                                                     [-0.4, 0, 0, 0],
                                                     [-0.5, 0, 0, 0]])*effect_size_factor,
                                            np.array([[0.6, 0.3, 0, 0],
                                                      [-0.4, 0, 0, 0],
                                                      [-0.5, 0, 0, 0]])*-1.0*effect_size_factor
                                            ]
                                 },
                    'cluster2': {'n': 25, 'changepoints': [9, 17],
                                 'state': [np.array([[10, 0.4, -0.04, 0.1],
                                                    [11, -0.4, 0.05, 0.4],
                                                    [1.2, -0.02, 0.03, 0.8]]),
                                           np.array([[10, 0.35, -0.04, 0.1],
                                                     [11.5, -0.3, 0.2, 0.3],
                                                     [1.2, -0.04, 0.03, 0.6]]),
                                           np.array([[10, 0.25, -0.01, 0.1],
                                                     [11.5, -0.2, 0.2, 0.3],
                                                     [1.2, -0.05, 0.03, 0.5]]),
                                           ],
                                 'action': [np.array([[0.6, 0.3, 0, 0],
                                                     [-0.4, 0, 0, 0],
                                                     [-0.5, 0, 0, 0]])*effect_size_factor,
                                            np.array([[0.6, 0.3, 0, 0],
                                                      [-0.4, 0, 0, 0],
                                                      [-0.5, 0, 0, 0]])*effect_size_factor,
                                            np.array([[0.6, 0.3, 0, 0],
                                                      [-0.4, 0, 0, 0],
                                                      [-0.5, 0, 0, 0]])*-1.0*effect_size_factor
                                            ]
                                 },
                    'cluster3': {'n': 25, 'changepoints': [26],
                                 'state': [np.array([[10, 0.4, -0.04, 0.1],
                                                    [11, -0.4, 0.05, 0.4],
                                                    [1.2, -0.02, 0.03, 0.8]])]*2,
                                 'action': [np.array([[0.6, 0.3, 0, 0],
                                                     [-0.4, 0, 0, 0],
                                                     [-0.5, 0, 0, 0]])*effect_size_factor]*2
                                 }
                    }


# importlib.reload(sim)
sim_dat = sim.simulate_data(T, cluster_settings)
States, Rewards, Actions = sim_dat.simulate(seed, burnin=60)

# make a copy
States_s = copy(States)
# dimension of States
p_var = 3
def transform(x):
    return (x - np.mean(x)) / np.std(x)
for i in range(p_var):
    States_s[:,:,i] = transform(States[:,:,i])

# number of clusters
K = len(cluster_settings)
N = sim_dat.N


# %% evaluation function
def evaluate(changepoints_true, g_index, predict, N, T):
    '''
    g_index : predicted group index
    predict : predicted changepoints
    '''
    changepoint_err = np.mean(np.abs(predict - changepoints_true)/T)
    cluster_err = adjusted_rand_score(changepoints_true, g_index)
    return changepoint_err, cluster_err


test_setting = "cdist"
method = 'proposed'
# create folder under seed if not existing
path_name = 'data/'
if not os.path.exists(path_name):
    os.makedirs(path_name, exist_ok=True)
path_name += '20220701/'
if not os.path.exists(path_name):
    os.makedirs(path_name, exist_ok=True)
path_name += "sim_" + effect_size + "/"
if not os.path.exists(path_name):
    os.makedirs(path_name, exist_ok=True)
path_name += method + "/"
if not os.path.exists(path_name):
    os.makedirs(path_name, exist_ok=True)


# %% conditional distribution
# changepoints_true = np.ones([N, 1]) * 25
# changepoints_true[50:] = 20
changepoints_true = np.repeat([11,17,0], [25]*3, axis=0)
# cluster_true = np.repeat([0,1,2], [25]*3, axis=0)
# threshold_type = "permutation"
# threshold_type = "Chi2" #"permutation"
# C3_candidates = [0.5, 1.0, 2.0, 3.0]
# best_IC = -1e10
# best_C3 = C3_candidates[0]
# out = None
# importlib.reload(mean_detect)
out = mean_detect.fit(States_s, Actions, example=test_setting, C3=1.0, init="changepoints",
                      K=K, threshold_type=threshold_type, B = 1000,
                      seed=seed, nthread=num_threads, max_iter=15)
# print(out)

# print("best_C3 =", best_C3)


### need to deal with label switching
# number of individuals each cluster
n_k = 25
from scipy.optimize import linear_sum_assignment
cluster_index = out[1].squeeze()
cluster_centers_true = []
cluster_centers_est = []
for k in range(K):
    cluster_centers_true.append(np.mean(States[0 + k * n_k:n_k + k * n_k, :, :], axis=0))
    cluster_centers_est.append(np.mean(States[cluster_index == k, :, :], axis=0))
# initialize cost matrix
cost_matrix = np.zeros(shape=(K, K))
# for each cluster
for k in range(K):
    for j in range(K):
        cost_matrix[k, j] = -cluster_centers_true[k].flatten() @ cluster_centers_est[j].flatten()
        # cost_matrix[k,j] = np.corrcoef(cluster_centers_true[k].flatten(), cluster_centers_est[k].flatten())[1,0]
row_ind, col_ind = linear_sum_assignment(cost_matrix)
cluster_index = np.argsort(col_ind)[cluster_index]

changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), cluster_index, out[2].squeeze(), N, T)

iter_num = out[0]
save_data = {
    'clusters': cluster_index,
    "changepoints": out[2],
    "changepoint_err": changepoint_err,
    "cluster_err": cluster_err,
    "iter_num": iter_num}
file_name = path_name + "seed" + str(seed) + "_" + threshold_type + "_cpresult" + ".dat"
# file_name = path_name + "cpresult_" + "seed" + str(seed) + ".dat"
with open(file_name, 'wb') as f:
    pickle.dump(save_data, f)

print("Clusters:", out[1])
print("Changepoints:", out[2].flatten())

