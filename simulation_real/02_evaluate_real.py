'''
Simulate nonstationary time series data and apply Q-learning.
Data generation mechanism resembles the IHS data
'''
#!/usr/bin/python
import platform, sys, os, pickle, re
plat = platform.platform()
print(plat)
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
type_est = str(sys.argv[3])
# seed=27
# type_est='proposed'

# import importlib
# importlib.reload(sim)
# importlib.reload(mean_detect)
# importlib.reload(stat)
# importlib.reload(plotting)

startTime = datetime.now()
np.random.seed(seed)
print("seed =", seed)
threshold_type = "Chi2"

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
    effect_size_factor = 0.4
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

test_setting = "cdist"
# create folder under seed if not existing
path_name = 'data/'
path_name += '20220701/'
path_name += "sim_" + effect_size + "/"

changepoints_true = np.repeat([11,17,0], [25]*3, axis=0)


#%% compare optimal policies
gamma = 0.9
basemodel = DecisionTreeRegressor(random_state=seed)
# param_grid = {"max_depth": [3],#
#               "min_samples_leaf": [50, 60]}#
param_grid = {"max_depth": [3, 5, 6],#
              "min_samples_leaf": [30, 40, 50, 60]}#
metric = 'ls'
qmodel = 'polynomial'
degree = 1
cp_true = [11, 17, 0]
N_new = 200
T_total = 200  # + cp_true[k]
plot_value = seed <= 3

def estimate_value(States, Rewards, Actions, cluster_index, cluster_changepoints, param_grid, basemodel):
    # number of clusters
    K = len(cluster_changepoints)

    ## first need to estimate the optimal policy for each estimated cluster
    opt_policy_list = []
    for k in range(K):
        print("\ncluster", k)
        idx = cluster_index == k
        cp_k = int(cluster_changepoints[k])
        States_k = States[idx, cp_k:, :]
        Rewards_k = Rewards[idx, cp_k:]
        Actions_k = Actions[idx, cp_k:]
        # States = copy(States_k)
        # Rewards = copy(Rewards_k)
        # Actions = copy(Actions_k)
        out = select_model_cv(States_k, Rewards_k, Actions_k, param_grid, bandwidth=1.0, qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,nfold = 5, num_threads = num_threads, metric = metric)
        model = out['best_model']
        print(model)
        q_all = stat.q_learning(States_k, Rewards_k, Actions_k, qmodel, degree, gamma, rbf_dim=degree, rbf_bw=1.0)
        q_all_fit = q_all.fit(model, max_iter=200, tol = 1e-6)
        opt_policy_list.append(q_all)

    ### now simulate new trajectories for each individual using their true dynamics
    # for each original individual, we simulate 8 new episodes
    N_trajectory_per_person = 8
    # N_individual_per_cluster = 25
    estimated_value_list = []
    cum_N = 0
    for k in range(K):
        cluster_k_settings = {}
        cluster_k_settings['cluster1'] = cluster_settings['cluster' + str(k+1)]
        cluster_k_settings['cluster1']['n'] = N_trajectory_per_person
        idx = cluster_index == k
        N_individual_in_cluster = sum(idx)
        cp_k = int(cluster_changepoints[k])
        for i in range(N_individual_in_cluster):
            estimated_cluster = cluster_index[i + cum_N]
            # get the policy for this cluster
            opt_policy = opt_policy_list[estimated_cluster]
            sim_dat = sim.simulate_data(T_total, cluster_k_settings)
            new_seed = seed*np.random.randint(10000)+np.random.randint(10000)*i+500*k
            States_new, Rewards_new, Actions_new = sim_dat.simulate(new_seed, T0=cp_k, T_total=T_total, burnin=00,
                                                                    optimal_policy_model=opt_policy)
            estimated_value = 0.0
            for t in range(T_total):
                estimated_value += Rewards_new[:, t] * gamma ** t
            estimated_value_list.append(estimated_value)
        cum_N += N_individual_in_cluster


    #
    # # a list of mean values for each clusters
    # estimated_value_list = []
    # for k in range(K):
    #     print("\ncluster", k)
    #     idx = cluster_index == k
    #     cp_k = int(cluster_changepoints[k])
    #     States_k = States[idx, cp_k:, :]
    #     Rewards_k = Rewards[idx, cp_k:]
    #     Actions_k = Actions[idx, cp_k:]
    #     # States = copy(States_k)
    #     # Rewards = copy(Rewards_k)
    #     # Actions = copy(Actions_k)
    #     out = select_model_cv(States_k, Rewards_k, Actions_k, param_grid, bandwidth=1.0, qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,nfold = 5, num_threads = num_threads, metric = metric)
    #     model = out['best_model']
    #     print(model)
    #     q_all = stat.q_learning(States_k, Rewards_k, Actions_k, qmodel, degree, gamma, rbf_dim=degree, rbf_bw=1.0)
    #     q_all_fit = q_all.fit(model, max_iter=200, tol = 1e-6)
    #     cluster_k_settings = {}
    #     cluster_k_settings['cluster1'] = cluster_settings['cluster' + str(k+1)]
    #     cluster_k_settings['cluster1']['n'] = N_new
    #     # importlib.reload(sim)
    #     sim_dat = sim.simulate_data(T_total, cluster_k_settings)
    #     States_new, Rewards_new, Actions_new = sim_dat.simulate(seed, T0 = cp_k, T_total = T_total, burnin=00, optimal_policy_model=q_all)
    #     if plot_value:
    #         fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    #         # plt.subplots_adjust(hspace=5)
    #         for a in range(2):
    #             tree.plot_tree(q_all.q_function_list[a], ax=axs[a])
    #             axs[a].set_title('Action ' + str(2*a-1), loc='left')
    #         # plt.tight_layout()
    #         fig.savefig('plot_policy' + type_est + '.pdf', bbox_inches='tight', pad_inches = 0.5)
    #         plt.close('all')
    #         # plt.show()
    #     estimated_value = 0.0
    #     for t in range(T_total):
    #         estimated_value += Rewards_new[:,t] * gamma**t
    #     estimated_value_list.append(np.mean(estimated_value))
    return estimated_value_list



#%% proposed method
if type_est == 'proposed':
    file_path = path_name + type_est + "/"
    # read in cp/clustering result
    saved_data = pickle.load(open(file_path + "seed" + str(seed) + "_" + threshold_type + "_cpresult" + ".dat", "rb"))
    # find the change point for each cluster
    cluster_cp = pd.crosstab(saved_data['clusters'], saved_data['changepoints'].flatten())
    cols = cluster_cp.columns
    bt = cluster_cp.apply(lambda x: x > 0)
    cluster_cp = np.concatenate(bt.apply(lambda x: list(cols[x.values]), axis=1))
    cluster_index = saved_data['clusters']
    cluster_changepoints = cluster_cp
    value = estimate_value(States, Rewards, Actions, cluster_index, cluster_changepoints, param_grid, basemodel)

    file_name = file_path + "seed" + str(seed) + "_value.dat"
    with open(file_name, 'wb') as f:
        pickle.dump(value, f)



#%% overall method: assume stationarity and homogeneity
if type_est == 'overall':
    file_path = path_name + type_est + "/"
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    cluster_changepoints = [0]
    cluster_index = np.zeros((N,), dtype='int32')
    # changepoints_overall = np.zeros(N, dtype='int32')
    # # clustering using all data
    # cluster_index = mean_detect.gmr(States_s, N, T, K, changepoints_overall, Actions, g_index=None, max_iter_gmr = 50)[0]
    # print("cluster_index=", cluster_index)
    #
    # # need to deal with label switching
    # from scipy.optimize import linear_sum_assignment
    # cluster_centers_true = []
    # cluster_centers_est = []
    # for k in range(K):
    #     cluster_centers_true.append(np.mean(States_s[0+k*25:25+k*25,:,:], axis = 0))
    #     cluster_centers_est.append(np.mean(States_s[cluster_index==k,:,:], axis = 0))
    # # initialize cost matrix
    # cost_matrix = np.zeros(shape = (K, K))
    # # for each cluster
    # for k in range(K):
    #     for j in range(K):
    #         cost_matrix[k,j] = -cluster_centers_true[k].flatten() @ cluster_centers_est[j].flatten()
    #         # cost_matrix[k,j] = np.corrcoef(cluster_centers_true[k].flatten(), cluster_centers_est[k].flatten())[1,0]
    # row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # cluster_index = np.argsort(col_ind)[cluster_index]
    #
    # print("cluster_index=", cluster_index)
    # # make the change points 0 for all clusters
    # cluster_changepoints = np.zeros(len(np.unique(cluster_index)))
    value = estimate_value(States, Rewards, Actions, cluster_index, cluster_changepoints, param_grid, basemodel)

    file_name = file_path + "seed" + str(seed) + "_value" + ".dat"
    with open(file_name, 'wb') as f:
        pickle.dump(value, f)



#%% true cluster method: assume true cluster membership is known, and are all homogenous
if type_est == 'truecluster':
    file_path = path_name + type_est + "/"
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    # cluster_changepoints = np.repeat([11,17,0], [25]*3, axis=0)
    # cluster_index = np.zeros((N,), dtype='int32')
    cluster_changepoints = [0,0,0]
    cluster_index = np.repeat([0,1,2], [25]*3, axis=0)
    value = estimate_value(States, Rewards, Actions, cluster_index, cluster_changepoints, param_grid, basemodel)
    file_name = file_path + "seed" + str(seed) + "_value" + ".dat"
    with open(file_name, 'wb') as f:
        pickle.dump(value, f)


#%% estimated cluster method: estimate cluster membership but assume homogeneity
if type_est == 'estimatedcluster':
    file_path = path_name + type_est + "/"
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    # clustering using all data
    changepoints_overall = np.zeros((N,), dtype='int32')
    cluster_index = mean_detect.gmr(States_s, N, T, K, changepoints_overall, Actions, g_index=None, max_iter_gmr = 50)[0]
    print("cluster_index=", cluster_index)

    # need to deal with label switching
    from scipy.optimize import linear_sum_assignment
    cluster_centers_true = []
    cluster_centers_est = []
    for k in range(K):
        cluster_centers_true.append(np.mean(States_s[0+k*25:25+k*25,:,:], axis = 0))
        cluster_centers_est.append(np.mean(States_s[cluster_index==k,:,:], axis = 0))
    # initialize cost matrix
    cost_matrix = np.zeros(shape = (K, K))
    # for each cluster
    for k in range(K):
        for j in range(K):
            cost_matrix[k,j] = -cluster_centers_true[k].flatten() @ cluster_centers_est[j].flatten()
            # cost_matrix[k,j] = np.corrcoef(cluster_centers_true[k].flatten(), cluster_centers_est[k].flatten())[1,0]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_index = np.argsort(col_ind)[cluster_index]
    print("cluster_index=", cluster_index)
    # make the change points 0 for all clusters
    cluster_changepoints = np.zeros(len(np.unique(cluster_index)))

    value = estimate_value(States, Rewards, Actions, cluster_index, cluster_changepoints, param_grid, basemodel)
    file_name = file_path + "seed" + str(seed) + "_value" + ".dat"
    with open(file_name, 'wb') as f:
        pickle.dump(value, f)



#%% oracle method: assume true cluster membership and true change points
if type_est == 'oracle':
    file_path = path_name + type_est + "/"
    if not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)
    # cluster_changepoints = np.repeat([11,17,0], [25]*3, axis=0)
    # cluster_index = np.zeros((N,), dtype='int32')
    cluster_changepoints = [11,17,0]
    cluster_index = np.repeat([0,1,2], [25]*3, axis=0)
    value = estimate_value(States, Rewards, Actions, cluster_index, cluster_changepoints, param_grid, basemodel)

    file_name = file_path + "seed" + str(seed) + "_value" + ".dat"
    with open(file_name, 'wb') as f:
        pickle.dump(value, f)