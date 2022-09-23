'''
Simulate nonstationary time series data and apply Q-learning.
Data generation mechanism resembles the IHS data
'''
#!/usr/bin/python
import platform, sys, os, pickle, re
plat = platform.platform()
# print(plat)
if plat == 'macOS-12.5-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private/simulation_real")
    sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")
elif plat == 'Linux-3.10.0-1160.6.1.el7.x86_64-x86_64-with-glibc2.17' or plat == 'Linux-3.10.0-1160.45.1.el7.x86_64-x86_64-with-glibc2.17':  # greatlakes
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# import simu.simu_mean_detect_modified as mean_detect
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
# seed = int(sys.argv[1])
# effect_size = str(sys.argv[2])
# # threshold_type = str(sys.argv[3]) #"Chi2" or "permutation"
seed = 90
effect_size = "strong"

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
if effect_size == "weak":
    effect_size_factor = 0.1
elif effect_size == "moderate":
    effect_size_factor = 0.3
elif effect_size == "strong":
    effect_size_factor = 1.0
base_transition = np.array([[10, 0.4, -0.04, 0.1],
                            [11, -0.4, 0.05, 0.4],
                            [1.2, -0.02, 0.03, 0.8]])
cluster_settings = {'cluster1': {'n': 25, 'changepoints': [11],
                                 'state': [base_transition +
                                           np.zeros(shape = (3,4)) * effect_size_factor,
                                           ]*2,
                                 'action': [np.array([[0.6, 0.3, 0, 0],
                                                     [-0.4, 0, 0, 0],
                                                     [-0.5, 0, 0, 0]])*effect_size_factor,
                                            np.array([[0.6, 0.3, 0, 0],
                                                      [-0.4, 0, 0, 0],
                                                      [-0.5, 0, 0, 0]])*-1.0*effect_size_factor
                                            ]
                                 },
                    'cluster2': {'n': 25, 'changepoints': [9, 17],
                                 'state': [base_transition +
                                           np.array([[-1.0, 0.0, 0, 0],
                                                    [-0.5, 0.0, 0.0, 0.0],
                                                    [-0.2, 0, 0.0, 0.0]]) * effect_size_factor,
                                           base_transition +
                                           np.array([[1.0, 0.15, -0.01, 0.02],
                                                     [1.0, -0.15, 0.01, 0.1],
                                                     [0.3, -0.01, 0.01, -0.15]]) * effect_size_factor,
                                           base_transition +
                                           np.array([[1.0, 0.15, -0.01, 0.02],
                                                     [1.0, -0.15, 0.01, 0.1],
                                                     [0.3, -0.01, 0.01, -0.15]]) * effect_size_factor
                                           ],
                                 'action': [np.array([[0.5, 0.3, 0, 0],
                                                     [-0.3, 0, 0, 0],
                                                     [-0.4, 0, 0, 0]])*effect_size_factor,
                                            np.array([[0.7, 0.2, 0, 0],
                                                      [-0.5, 0, 0, 0],
                                                      [-0.6, 0, 0, 0]])*-1.0*effect_size_factor,
                                            np.array([[0.7, 0.2, 0, 0],
                                                      [-0.5, 0, 0, 0],
                                                      [-0.6, 0, 0, 0]])*effect_size_factor
                                            ]
                                 },
                    'cluster3': {'n': 25, 'changepoints': [26],
                                 'state': [base_transition +
                                           np.array([[1.5, 0.05, 0, 0],
                                                     [0.5, -0.2, 0.0, 0.0],
                                                     [0.2, 0, 0.0, 0.0]]) * effect_size_factor
                                           ]*2,
                                 'action': [np.array([[0.55, 0.25, 0, 0],
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
def evaluate(changepoints_true, g_index, predict, T):
    '''
    g_index : predicted group index
    predict : predicted changepoints
    '''
    changepoint_err = np.mean(np.abs(predict - changepoints_true)/T)
    cluster_err = adjusted_rand_score(g_index_true, g_index)
    return changepoint_err, cluster_err

def evaluate_Klist(K_list, out, changepoints_true, T):
    changepoint_err_list = [None] * len(K_list)
    cluster_err_list = [None] * len(K_list)
    for k in K_list:
        tmp = out.models[k]
        # tmp = out.models[K_list.index(k)]
        changepoint_err_list[K_list.index(k)], cluster_err_list[K_list.index(k)] = evaluate(changepoints_true,
                                                                                            tmp.g_index,
                                                                                            tmp.changepoints, T)
    return np.array(changepoint_err_list), np.array(cluster_err_list)

changepoints_true = np.repeat([11,17,0], [25]*3, axis=0)
g_index_true = np.repeat([0,1,2], [25]*3, axis=0)


#%% simu0: KMeans clustering + multiple random change points
setting = ""
print(setting)
K_list = range(2, 7)

from tslearn.clustering import TimeSeriesKMeans

# perform time series K means for all candidate K's
g_index_init_list = []
for k in K_list:
    model = TimeSeriesKMeans(n_clusters=k, metric="euclidean", max_iter=50,
                             random_state=0).fit(States_s)
    g_index_init_list.append(model.labels_)

K_list = [3]
g_index_init_list = [g_index_true]
# number of random initial changepoints
epsilon = 0.05
n_init_cp = 3
best_model_IC = -1e9
best_model = None
changepoints_init = np.random.choice(range(T - 1), size=N)
# importlib.reload(mean_detect)
out = mean_detect.fit_tuneK(K_list, States_s, Actions, init="clustering", example="cdist",
                            seed=seed, nthread=num_threads, max_iter=15, threshold_type="permutation",
                            C=1.0, changepoints_init=changepoints_init, g_index_init_list=g_index_init_list,
                            clustering_warm_start=1, distance_metric="euclidean", B = 100, epsilon = epsilon,
                            kappa_min=6, kappa_max=22, kappa_interval = 1)
model_IC = out[2][6]
print("model_IC =", model_IC)
changepoint_err, cluster_err = evaluate_Klist(K_list, out, changepoints_true, T)
print('changepoint error:', changepoint_err)
print('ARI:', cluster_err)
print("Clusters:", out[2][1])
print("Changepoints:", out[2][2])

