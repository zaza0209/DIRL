'''
Simulate nonstationary time series data and apply Q-learning.
Data generation mechanism resembles the IHS data
'''
#!/usr/bin/python
import platform, sys, os, pickle, re
from copy import deepcopy
from copy import copy
plat = platform.platform()
print(plat)
if plat == 'macOS-12.5-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private/simulation_real")
    sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")
elif plat == 'Linux-4.18.0-305.45.1.el8_4.x86_64-x86_64-with-glibc2.28':  # greatlakes
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
from simu.evaluation import *
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from functions.evaluation_separateA import *
# import HeterRL.functions.compute_test_statistics_separateA as stat


# Arguments passed

# argv should be: seed, kappa, degree, num_threads
seed = int(sys.argv[1])
effect_size = str(sys.argv[2])
# gamma = float(sys.argv[5])
type_est = str(sys.argv[3])
# # threshold_type = str(sys.argv[3]) #"Chi2" or "permutation"
# seed = 20
# effect_size = "strong"
# K = 3
# init = 20
gamma = 0.9
# type_est = 'proposed'

# import importlib
# importlib.reload(sim)
# importlib.reload(mean_detect)
# importlib.reload(stat)
# importlib.reload(plotting)

startTime = datetime.now()
np.random.seed(seed)
print("seed =", seed)
# threshold_type = "Chi2"

# %% simulate data
# terminal timestamp
T_initial = 26
N_per_cluster = 10
num_threads = 5
# dimension of states
p = 3
test_setting = "cdist"
# effect_size = "large"
if effect_size == "weak":
    effect_size_factor = 0.2
elif effect_size == "moderate":
    effect_size_factor = 0.5
elif effect_size == "strong":
    effect_size_factor = 0.8
base_transition = np.array([[10, 0.4, -0.04, 0.1],
                            [11, -0.4, 0.05, 0.4],
                            [1.2, -0.02, 0.03, 0.8]])
cluster_settings = {'cluster1': {'n': N_per_cluster, 'changepoints': [11],
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
                    'cluster2': {'n': N_per_cluster, 'changepoints': [9, 17],
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
                    'cluster3': {'n': N_per_cluster, 'changepoints': [26],
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
# simulate the first offline batch
sim_dat = sim.simulate_data(T_initial, cluster_settings)
States, Rewards, Actions = sim_dat.simulate(seed, burnin=0)
# number of clusters
K_true = len(cluster_settings)
N = sim_dat.N
def transform(x):
    return (x - np.mean(x)) / np.std(x)

#%% environment for saving online values
data_path = 'data/'
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
data_path += '20221106/'
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
data_path += "value_" + effect_size + "/"
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
data_path += "seed" + str(seed) + "/"
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)


# print(data_path)
# os.chdir(data_path)
# print("os.curdir =", os.curdir)
stdoutOrigin = sys.stdout
sys.stdout = open(data_path + "/log_online_" + type_est + ".txt", "w")


#%% setup for policy estimation
rbf_bw = 0.1
metric = 'kerneldist'
param_grid = {"max_depth": [3, 5],#
              "min_samples_leaf": [50, 60, 70]}#
basemodel = DecisionTreeRegressor(random_state=seed)
# model=DecisionTreeRegressor(random_state=10, max_depth=3, min_samples_leaf = 60)
T_new = 100+T_initial
cp_arrival_mean = 25
# create a random list of change points
change_points_generated = np.random.poisson(cp_arrival_mean)
change_points_subsequent_clusters12 = {
    1: [11, T_initial + change_points_generated],
    2: [17, T_initial + change_points_generated]
}
change_point_next = change_points_subsequent_clusters12[1][-1]
# we do not want a change point in the last 10 time points
while change_point_next < T_new - 10:
    change_point_next += np.random.poisson(cp_arrival_mean)
    change_points_subsequent_clusters12[1].append(change_point_next)
    change_points_subsequent_clusters12[2].append(change_point_next)
change_points_subsequent_clusters12[1] = change_points_subsequent_clusters12[1][:-1]
change_points_subsequent_clusters12[2] = change_points_subsequent_clusters12[2][:-1]
print("change_points =", change_points_subsequent_clusters12)
change_points_subsequent_clusters12[1].append(T_new)
change_points_subsequent_clusters12[2].append(T_new)

epsilon = 0.05
best_model_IC = -1e9
best_model = None
nthread = 5
threshold_type = "maxcusum"
max_iter = 10
B = 5000
qmodel = 'polynomial'
degree = 1
num_threads = 1
changepoints_true = np.repeat([11,17,0], [N_per_cluster]*3, axis=0)
g_index_true = np.repeat([0,1,2], [N_per_cluster]*3, axis=0)

K_list = [2,3,4]
path_name = 'data/20221031_C0/sim_' + effect_size + "/"

def estimate_value(States, Rewards, Actions, type_est, param_grid, basemodel):
    method = ''.join([i for i in type_est if not i.isdigit()])

    cluster1_label = 2
    cluster2_label = 5
    cp_detect_interval = 20
    seed_new = seed
    np.random.seed(seed)
    cluster_settings_batch = deepcopy(cluster_settings)
    cluster_settings_batch['cluster2']['changepoints'] = [cluster_settings['cluster2']['changepoints'][1]]
    cluster_settings_batch['cluster2']['state'] = [cluster_settings['cluster2']['state'][1], cluster_settings['cluster2']['state'][2]]
    cluster_settings_batch['cluster2']['action'] = [cluster_settings['cluster2']['action'][1], cluster_settings['cluster2']['action'][2]]

    States_updated = copy(States)
    Rewards_updated = copy(Rewards)
    Actions_updated = copy(Actions)

    # number of data batches
    n_batch = int((T_new - T_initial) / cp_detect_interval)
    cp_index = 0
    cp_current = np.repeat(0, N)
    T_current = T_initial
    T_length = T_initial

    for batch_index in range(n_batch):

        ### Proposed method
        if method == "proposed":
            if batch_index == 0:
                cp_current = np.repeat(0,N)
                T_length = T_initial
                kappa_min = int((T_length - 1 - 17) * 0.8)
                kappa_max = min(T_length - 3, int((T_length - 1) * 1.2))

                # read the recorded results from csv
                initial_result = pd.read_csv("output/20221031_C0_offline_summarized.csv")
                selected_model = initial_result[(initial_result['seed'] == seed) &
                               (initial_result['Effect Size'] == effect_size)]
                selected_K = selected_model['K'].iloc[0]
                selected_init = selected_model['init'].iloc[0]
                file_name = path_name + "result_seed" + str(seed) + "_K" + str(selected_K) + "_init" + str(selected_init) + ".dat"
                out = pickle.load(open(file_name, "rb"))
                cp_current = out['changepoints']
                g_index = out['clusters']

            else:
                T_length = States_updated[:, int(np.max(cp_current)):, :].shape[1] - 1
                kappa_min = 5
                kappa_max = min(T_length - 3, 30)
                States_s = copy(States_updated[:, int(np.max(cp_current)):, :])
                for i in range(3):
                    States_s[:, :, i] = transform(States_s[:, :, i])
                epsilon = 1/T_length

                # init with a fixed change point
                changepoints_init = np.repeat(10, N)
                out1 = mean_detect.fit_tuneK(K_list, States_s, Actions_updated[:, int(np.max(cp_current)):],
                                     seed = seed, init = "changepoints", changepoints_init=changepoints_init, B=B,
                                     epsilon=epsilon, nthread=nthread,
                                     kappa_min = kappa_min, kappa_max = kappa_max, max_iter=max_iter,
                                     init_cluster_range = T_length-1-kappa_min, threshold_type=threshold_type,
                                     is_cp_parallel=1, is_tune_parallel=1)
                IC1 = out1[1]
                result = out1[2]
                cp_current = np.repeat(int(np.max(cp_current)), N) + result[2].flatten()
                g_index = result[1]
            print('T_length', T_length, ', cp_current, ',cp_current)

        ### Overall method
        if method == "overall":
            # if T_current > 200 and T_current % 100 == 0:
            #     cp_current = np.repeat(max(0, T_current - 200), N)
            print("cp_current =", cp_current)
            g_index = np.repeat(0, N)

        ### Oracle method
        if method == "oracle":
            if batch_index == 0:
                cp_current = changepoints_true# np.minimum(changepoints_true, np.repeat(T_current, N))
                g_index = g_index_true
            else:
                if batch_index == 0:
                    cp_current = np.minimum(T_current, change_points_subsequent_clusters12[1][cp_index])
                else:
                    cp_current = np.maximum(cp_current, change_points_subsequent_clusters12[1][cp_index])

                g_index = g_index_true


        ### Clustering only method
        if method == "cluster_only":
            if batch_index != 0:
                T_length = States_updated[:, int(cp_current[0]):, :].shape[1]
            print('T_length =', T_length, ', cp_current =', cp_current)
            States_s = copy(States_updated[:, int(cp_current[0]):, :])
            for i in range(p):
                States_s[:, :, i] = transform(States_s[:, :, i])
            out = mean_detect.fit_tuneK(K_list, States_s, Actions_updated[:,  int(cp_current[0]):],
                                 seed = seed+batch_index, init = "changepoints", nthread=nthread,changepoints_init =cp_current,
                                 max_iter=1,is_only_cluster=1, is_tune_parallel=1, C=0)
            best_out = out.best_model
            g_index = best_out[1]

        ### Change point only method
        if method == "changepoint_only":
            g_index = np.repeat(0, N)
            if batch_index == 0:
                kappa_min = int((T_length - 1 - 17) * 0.8)
                kappa_max = min(T_length - 3, int((T_length - 1) * 1.2))
            else:
                T_length = States_updated[:, int(np.max(cp_current)):, :].shape[1]  # - 1
                kappa_min = 5
                kappa_max = min(T_length - 3, 30)
            States_s = copy(States_updated[:, int(np.max(cp_current)):, :])
            for i in range(p):
                States_s[:, :, i] = transform(States_s[:, :, i])
            epsilon = 1 / T_length
            out = mean_detect.fit_tuneK([1], States_s, Actions_updated[:, int(np.max(cp_current)):],
                                        seed=seed + batch_index, init="clustering", epsilon=epsilon, nthread=nthread,
                                        kappa_min=kappa_min, kappa_max=kappa_max, max_iter=1,
                                        g_index_init_list=[g_index], C=0,
                                        is_tune_parallel=0, is_cp_parallel=1)
            cp_current += out.best_model[2]


        #%% estimate the optimal policy
        q_all_group = [None]*len(set(g_index))
        q_all_fit = [None]*len(set(g_index))
        print('States_updated.shape', States_updated.shape)
        policyTime = datetime.now()
        for k in np.unique(g_index):
            print('k =', k, ', batch_index =', batch_index)
            # if batch_index % 2 == 0:
            # print('States_updated[g_index == ',g,', ', cp_current[np.where(g_index==g)[0][0]], ':, :]',States_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:, :].shape)
            try:
                cluster_k_idx = (g_index == k)
                cp_idx = np.where(g_index==k)[0][0]
                N_k = np.sum(cluster_k_idx)
                if batch_index % 2 == 0:
                    out = select_model_cv(States_updated[cluster_k_idx, cp_current[cp_idx]:, :].reshape((N_k, -1, p)),
                                          Rewards_updated[cluster_k_idx, cp_current[cp_idx]:].reshape((N_k, -1)),
                                          Actions_updated[cluster_k_idx, cp_current[cp_idx]:].reshape((N_k, -1)),
                                          param_grid, bandwidth=rbf_bw,
                                          qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                                          nfold = 5, num_threads = nthread, metric = metric)
                    model = out['best_model']
                    print(model)
                q_all_group[k] = stat.q_learning(States_updated[cluster_k_idx, cp_current[cp_idx]:, :],
                                                 Rewards_updated[cluster_k_idx, cp_current[cp_idx]:],
                                                 Actions_updated[cluster_k_idx, cp_current[cp_idx]:], qmodel, degree, gamma, rbf_dim=degree, rbf_bw=rbf_bw)
                q_all_fit[k] = q_all_group[k].fit(model, max_iter=200, tol = 1e-6)
            except:
                print('!!! BUG in policy learning')
                with open('bugdata.dat' , "wb") as f:
                    pickle.dump({'States':States_updated[cluster_k_idx, cp_current[cp_idx]:, :].reshape((N_k, -1, p)),
                                 'Actions':Actions_updated[cluster_k_idx, cp_current[cp_idx]:].reshape((N_k, -1)),
                                 'Rewards':Rewards_updated[cluster_k_idx, cp_current[cp_idx]:].reshape((N_k, -1))}, f)
                exit()

        #%% now we collect the next batch of data following the estimated policy
        T_current += cp_detect_interval
        print("batch_index =", batch_index, "cp_index =", cp_index, "cp_current =", cp_current, "T_to_simulate =", T_current)
        cluster_settings_batch['cluster3']['changepoints'] = [cp_detect_interval]

        # if the next change point has not been encountered, just keep the current dynamics
        if T_current <= change_points_subsequent_clusters12[1][cp_index+1]:
            print("same as before")
            # system_settings_batch['changepoints'] = [T_current]
            for k in [1, 2]:
                cluster_settings_batch['cluster'+str(k)]['changepoints'] = [cp_detect_interval]
                cluster_settings_batch['cluster'+str(k)]['state'] = [cluster_settings_batch['cluster'+str(k)]['state'][1]] * 2
                cluster_settings_batch['cluster'+str(k)]['action'] = [cluster_settings_batch['cluster'+str(k)]['action'][1]] * 2

        # if the next change point is encountered, need to make it the change point
        else:
            cluster_settings_batch['cluster1']['changepoints'] = [change_points_subsequent_clusters12[1][cp_index+1] - T_current + cp_detect_interval]
            cluster_settings_batch['cluster2']['changepoints'] = [change_points_subsequent_clusters12[1][cp_index+1] - T_current + cp_detect_interval]
            np.random.seed(seed*batch_index)

            #%% check if clusters 1 and 2 share the same dynamics. if yes, then split
            if cluster1_label == cluster2_label == 2:
                print("split")
                # then we will split, with equal probabilities being split into either dynamic
                if np.random.rand() < 0.5:
                    ## 25
                    ## 22
                    cluster_settings_batch['cluster1']['state'][0] = cluster_settings_batch['cluster1']['state'][1]
                    cluster_settings_batch['cluster1']['state'][1] = cluster_settings['cluster2']['state'][2]
                    cluster_settings_batch['cluster2']['state'] = [cluster_settings_batch['cluster2']['state'][1]]*2
                    cluster_settings_batch['cluster1']['action'][0] = cluster_settings_batch['cluster1']['action'][1]
                    cluster_settings_batch['cluster1']['action'][1] = cluster_settings['cluster2']['action'][2]
                    cluster_settings_batch['cluster2']['action'] = [cluster_settings_batch['cluster2']['action'][1]]*2
                    cluster1_label = 5
                else:
                    ## 22
                    ## 25
                    cluster_settings_batch['cluster1']['state'] = [cluster_settings_batch['cluster1']['state'][1]]*2
                    cluster_settings_batch['cluster2']['state'][0] = cluster_settings_batch['cluster2']['state'][1]
                    cluster_settings_batch['cluster2']['state'][1] = cluster_settings['cluster2']['state'][2]
                    cluster_settings_batch['cluster1']['action'] = [cluster_settings_batch['cluster1']['action'][1]]*2
                    cluster_settings_batch['cluster2']['action'][0] = cluster_settings_batch['cluster2']['action'][1]
                    cluster_settings_batch['cluster2']['action'][1] = cluster_settings['cluster2']['action'][2]
                    cluster2_label = 5
            # check if clusters 1 and 2 share the same dynamics
            elif cluster1_label == cluster2_label == 5:
                print("split")
                # then we will split, with equal probabilities being split into either dynamic
                if np.random.rand() < 0.5:
                    ## 52
                    ## 55
                    cluster_settings_batch['cluster1']['state'][0] = cluster_settings_batch['cluster1']['state'][2]
                    cluster_settings_batch['cluster1']['state'][1] = cluster_settings['cluster1']['state'][1]
                    cluster_settings_batch['cluster2']['state'] = [cluster_settings_batch['cluster2']['state'][1]]*2
                    cluster_settings_batch['cluster1']['action'][0] = cluster_settings_batch['cluster1']['action'][2]
                    cluster_settings_batch['cluster1']['action'][1] = cluster_settings['cluster1']['action'][1]
                    cluster_settings_batch['cluster2']['action'] = [cluster_settings_batch['cluster2']['action'][1]]*2
                    cluster1_label = 2
                else:
                    ## 55
                    ## 52
                    cluster_settings_batch['cluster1']['state'] = [cluster_settings_batch['cluster1']['state'][1]]*2
                    cluster_settings_batch['cluster2']['state'][0] = cluster_settings_batch['cluster2']['state'][1]
                    cluster_settings_batch['cluster2']['state'][1] = cluster_settings['cluster2']['state'][2]
                    cluster_settings_batch['cluster1']['action'] = [cluster_settings_batch['cluster1']['action'][1]]*2
                    cluster_settings_batch['cluster2']['action'][0] = cluster_settings_batch['cluster2']['action'][1]
                    cluster_settings_batch['cluster2']['action'][1] = cluster_settings['cluster2']['action'][2]
                    cluster2_label = 2


            #%% check if clusters 1 and 2 share the same dynamics. if not, then merge or evolution+constancy
            else: # merge
                print("merge")
                # with 0.5 probability to merge
                if np.random.rand() < 0.5:
                    # cluster 1 merges into cluster 2
                    cluster_settings_batch['cluster1']['state'][1] = cluster_settings_batch['cluster2']['state'][0]
                    cluster_settings_batch['cluster2']['state'][1] = cluster_settings_batch['cluster2']['state'][0]
                    cluster_settings_batch['cluster1']['action'][1] = cluster_settings_batch['cluster2']['action'][0]
                    cluster_settings_batch['cluster2']['action'][1] = cluster_settings_batch['cluster2']['action'][0]
                    cluster1_label = cluster2_label
                else:
                    # cluster 2 merges into cluster 1
                    cluster_settings_batch['cluster1']['state'][1] = cluster_settings_batch['cluster1']['state'][0]
                    cluster_settings_batch['cluster2']['state'][1] = cluster_settings_batch['cluster1']['state'][0]
                    cluster_settings_batch['cluster1']['action'][1] = cluster_settings_batch['cluster1']['action'][0]
                    cluster_settings_batch['cluster2']['action'][1] = cluster_settings_batch['cluster1']['action'][0]
                    cluster2_label = cluster1_label
            # # evolution+constancy
            # else:

            print("change point encountered")
            cp_index += 1
        print(cluster_settings_batch, "\n")


        # simulate new data following the estimated policy
        # print(States_new[199,0:,0])
        # print(Rewards_new[199,0:])
        # print(Actions_new[199,0:])
        # print(States_updated[3,100:,0])
        # print(Rewards_updated[5,100:])
        # print(Actions_updated[5,100:])

        seed_new = int(np.sqrt(np.random.randint(1e6) + seed_new*np.random.randint(10)))
        S0 = States_updated[:, -1, :]
        sim_dat = sim.simulate_data(cp_detect_interval, cluster_settings_batch)
        States_new, Rewards_new, Actions_new = sim_dat.simulate(seed_new, burnin=0, S0 = S0, optimal_policy_model=q_all_group)

        States_updated = np.concatenate((States_updated, States_new[:,1:,:]), axis = 1)
        Rewards_updated = np.concatenate((Rewards_updated, Rewards_new), axis = 1)
        Actions_updated = np.concatenate((Actions_updated, Actions_new), axis = 1)

        sys.stdout.flush()

    # %% compute values
    values = {}
    # discounted reward
    estimated_value = 0.0
    for t in range(T_initial, Rewards_updated.shape[1]):
        estimated_value += Rewards_updated[:, t] * gamma ** (t - T_initial)
    values['discounted_reward'] = np.mean(estimated_value)
    values['average_reward'] = np.mean(Rewards_updated[:, T_initial:])
    values['raw_reward'] = Rewards_updated
    return values



# %% run the evaluation
# method_list = ['oracle', 'proposed', 'overall', 'random', 'kernel0', 'kernel01', 'kernel02', 'kernel03', 'kernel04']
value = estimate_value(States, Rewards, Actions, type_est=type_est, param_grid=param_grid, basemodel=basemodel)
print(type_est, "discounted reward:", value['discounted_reward'], "\n")
print(type_est, "average reward:", value['average_reward'], "\n")
with open(data_path + "/value_online_" + type_est + ".dat", "wb") as f:
    pickle.dump(value, f)


sys.stdout.flush()
print('Finished. Time: ', datetime.now() - startTime)

sys.stdout.close()
sys.stdout = stdoutOrigin
