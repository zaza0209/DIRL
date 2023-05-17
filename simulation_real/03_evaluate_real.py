'''
Simulate nonstationary time series data and apply Q-learning.
Data generation mechanism resembles the IHS data
'''
#!/usr/bin/python
import platform, sys, os, pickle, re
from copy import deepcopy
from copy import copy
plat = platform.platform()
# print(plat)
if plat == 'macOS-13.0-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private/simulation_real")
    sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")
elif plat == 'Linux-4.18.0-305.65.1.el8_4.x86_64-x86_64-with-glibc2.28':  # greatlakes
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")

import numpy as np
from datetime import datetime
import simu.simu_mean_detect as mean_detect
import simulation_real.simulate_data_real as sim
from copy import copy
from copy import deepcopy
import pandas as pd
# from functions.evaluation import *
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
# from functions.utilities import *
from functions.evaluation_separateA import *
import functions.compute_test_statistics_separateA as stat


# Arguments passed

# argv should be: seed, kappa, degree, num_threads
seed = int(sys.argv[1])
effect_size = str(sys.argv[2])
# gamma = float(sys.argv[5])
type_est = str(sys.argv[3])
# # threshold_type = str(sys.argv[3]) #"Chi2" or "permutation"
# seed = 3
# effect_size = "strong"
# type_est = 'oracle'

# import importlib
# importlib.reload(sim)
# importlib.reload(mean_detect)

#%% environment for saving online values
data_path = 'data/'
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
data_path += '20230510/'
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
data_path += "value_" + effect_size + "/"
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
data_path += "seed" + str(seed) + "/"
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
stdoutOrigin = sys.stdout
sys.stdout = open(data_path + "/log_online_" + type_est + ".txt", "w")


#%%
startTime = datetime.now()
np.random.seed(seed)
print("seed =", seed)
# threshold_type = "Chi2"
gamma = 0.9

# %% simulate data
# terminal timestamp
T_initial = 26
N_per_cluster = 50
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

# base_transition = np.array([[10, 0.1, -0.02, 0.1],
#                             [11, -0.1, 0.02, 0],
#                             [1.2, -0.02, 0.01, 0]])
base_transition = np.array([[10, 0.4, -0.04, 0.1],
                            [11, -0.4, 0.05, 0.4],
                            [1.2, -0.02, 0.03, 0.8]])

def transition_function11(St, At, t):
    St_full = np.insert(St, 0, 1, axis=0)
    # print(St_full)
    return (base_transition + np.zeros(shape=(3, 4)) * effect_size_factor +\
            At * np.array([[0.6, 0.6, 0, 0],
                           [-0.4, 0.3, 0, 0],
                           [-0.5, -0.3, 0, 0]]) * effect_size_factor) @ St_full
def transition_function12(St, At, t):
    St_full = np.insert(St, 0, 1, axis=0)
    return (base_transition + np.zeros(shape=(3, 4)) * effect_size_factor +\
            (-1) * At * np.array([[0.6, 0.6, 0, 0],
                                  [-0.4, 0.3, 0, 0],
                                  [-0.5, -0.3, 0, 0]]) * effect_size_factor) @ St_full

def transition_function21(St, At, t):
    St_full = np.insert(St, 0, 1, axis=0)
    return (base_transition +\
            np.array([[-1.0, 0, 0, 0],
                      [-0.5, 0, 0.0, 0.0],
                      [-0.2, 0, 0.0, 0.0]]) * effect_size_factor +\
            At * np.array([[0.5, 0.3, 0, 0],
                           [-0.3, 0, 0, 0],
                           [-0.4, 0, 0, 0]]) * effect_size_factor) @ St_full
def transition_function22(St, At, t):
    St_full = np.insert(St, 0, 1, axis=0)
    return (base_transition + \
            np.array([[1.0, 0.15, -0.01, 0.02],
                      [1.0, -0.15, 0.01, 0.1],
                      [0.3, -0.01, 0.01, -0.15]]) * effect_size_factor +\
            (-1)*At * np.array([[0.7, 0.2, 0, 0],
                                [-0.5, 0, 0, 0],
                                [-0.6, 0, 0, 0]]) * effect_size_factor) @ St_full
def transition_function23(St, At, t):
    St_full = np.insert(St, 0, 1, axis=0)
    return (base_transition + \
            np.array([[1.0, 0.15, -0.01, 0.02],
                      [1.0, -0.15, 0.01, 0.1],
                      [0.3, -0.01, 0.01, -0.15]]) * effect_size_factor +\
            At * np.array([[0.7, 0.2, 0, 0],
                           [-0.5, 0, 0, 0],
                           [-0.6, 0, 0, 0]]) * effect_size_factor) @ St_full

def transition_function3(St, At, t):
    St_full = np.insert(St, 0, 1, axis=0)
    return (base_transition + \
            np.array([[1.5, 0.05, 0, 0],
                      [0.5, -0.2, 0.0, 0.0],
                      [0.2, 0, 0.0, 0.0]]) * effect_size_factor +\
            At * np.array([[0.55, 0.25, 0, 0],
                           [-0.4, 0, 0, 0],
                           [-0.5, 0, 0, 0]]) * effect_size_factor) @ St_full

def reward_function(St, At, t):
    return St[0]

system_settings_list_initial = [{'N': N_per_cluster, 'T': T_initial,
                         'changepoints': [11],
                         'state_functions': [transition_function11, transition_function12],
                         'reward_functions': reward_function
                         },
                        {'N': N_per_cluster, 'T': T_initial,
                         'changepoints': [9, 17],
                         'state_functions': [transition_function21, transition_function22, transition_function23],
                         'reward_functions': reward_function
                         },
                        {'N': N_per_cluster, 'T': T_initial,
                         'changepoints': [0],
                         'state_functions': [transition_function3, transition_function3],
                         'reward_functions': reward_function
                         }]

# number of clusters
K_true = len(system_settings_list_initial)

# simulate the first offline batch
States, Rewards, Actions = sim.simulate(system_settings=system_settings_list_initial[0], seed=seed)
for k in range(1, K_true):
    States_k, Rewards_k, Actions_k = sim.simulate(system_settings = system_settings_list_initial[k], seed = seed)
    States = np.concatenate((States, States_k), axis=0)
    Rewards = np.concatenate((Rewards, Rewards_k), axis=0)
    Actions = np.concatenate((Actions, Actions_k), axis=0)


# importlib.reload(sim)
N = States.shape[0]
def transform(x):
    return (x - np.mean(x)) / np.std(x)


# print(data_path)
# os.chdir(data_path)
# print("os.curdir =", os.curdir)


#%% setup for policy estimation

system_settings_list = [{'N': 1, 'T': T_initial,
                         'changepoints': [11],
                         'state_functions': [transition_function11, transition_function12],
                         'reward_functions': reward_function
                         },
                        {'N': 1, 'T': T_initial,
                         'changepoints': [17],
                         'state_functions': [transition_function22, transition_function23],
                         'reward_functions': reward_function
                         },
                        {'N': 1, 'T': T_initial,
                         'changepoints': [0],
                         'state_functions': [transition_function3, transition_function3],
                         'reward_functions': reward_function
                         }]

rbf_bw = 0.1
metric = 'kerneldist'
basemodel = DecisionTreeRegressor(random_state=seed)
N_new = N
T_new = 100+T_initial
cp_arrival_mean = 25
# create a random list of change points
changepoint_list = [11, 17, 0]
change_points_subsequent = [changepoint_list, T_initial + np.random.poisson(cp_arrival_mean)]
change_point_next = change_points_subsequent[1]
# we do not want a change point in the last 10 time points
while change_point_next < T_new - 10:
    change_point_next += np.random.poisson(cp_arrival_mean)
    change_points_subsequent.append(change_point_next)
change_points_subsequent = change_points_subsequent[:-1]
print("change_points =", change_points_subsequent)
change_points_subsequent.append(T_new)

epsilon = 0.05
best_model_IC = -1e9
best_model = None
nthread = 5
threshold_type = "maxcusum"
max_iter = 10
B = 5000
B_cp = 10000
qmodel = 'polynomial'
degree = 1
num_threads = 1
changepoints_true = np.repeat([11,17,0], [N_per_cluster]*3, axis=0)
g_index_true = np.repeat([0,1,2], [N_per_cluster]*3, axis=0)
cp_detect_interval = 10
system_indicator = [0,1,2]
K_list = [2,3,4]
param_grid_policy = {"max_depth": [3, 5],#
              "min_samples_leaf": [50, 60, 70]}#
param_grid_cp = {"alpha": [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
              "gamma": [3.7]}

print("States.shape =", States.shape)
# test=pickle.load(open("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private/simulation_real/bugdata_proposed.dat", "rb"))
# States=test['States']
# Rewards=test['Rewards']
# Actions=test['Actions']

def estimate_value(States, Rewards, Actions, type_est, param_grid, basemodel):
    system_list = []
    method = ''.join([i for i in type_est if not i.isdigit()])
    print('method', method)
    # cp_detect_interval = cp_detect_interval
    seed_new = seed
    np.random.seed(seed)
    system_settings_list[0]["T"] = cp_detect_interval
    system_settings_list[1]["T"] = cp_detect_interval
    system_settings_list[2]["T"] = cp_detect_interval

    States_updated = copy(States)
    Rewards_updated = copy(Rewards)
    Actions_updated = copy(Actions)

    # number of data batches
    n_batch = int((T_new - T_initial) / cp_detect_interval)
    cp_index = 0
    cp_current = np.repeat(0, N)
    T_current = T_initial
    T_length = T_initial
    # # first detect change point
    # if method == "random":
    #     cp_current = 0
    change_or_not = [None] * 2
    for batch_index in range(n_batch): #
        print('======= batch_index ======', batch_index)

        # %% first, detect change point
        # cp_current = change_points_subsequent[cp_index]
        cpTime = datetime.now()

        ### Overall method
        if method == "overall":
            # if T_current > 200 and T_current % 100 == 0:
            #     cp_current = np.repeat(max(0, T_current - 200), N)
            print("cp_current =", cp_current)
            g_index = np.repeat(0, N)

        ### Oracle method
        if method == "oracle":
            if batch_index == 0:
                cp_current = deepcopy(changepoints_true)  # np.minimum(changepoints_true, np.repeat(T_current, N))
                g_index = g_index_true
            else:
                # if batch_index == 1:
                #     cp_current = np.maximum(cp_current, changepoints_true)
                # else:
                print("change_or_not =", change_or_not)
                for g in range(2):
                    if change_or_not[g] != 0:
                        cp_current[g_index_true == g] = np.repeat(change_points_subsequent[cp_index], N_per_cluster)
                # if clusters 1 and 2 merge, then let the cluster index be 0, 0, 1
                if system_indicator[0] == system_indicator[1]:
                    g_index = np.repeat(0, N)
                    g_index[g_index_true == 2] = 1
                else:
                    g_index = g_index_true

            # print("batch_index =", batch_index)
            # print("changepoints_true =", changepoints_true)
            print("g_index =", g_index)

        ### Clustering only method
        if method == "cluster_only":
            # if T_current > 200 and T_current % 100 == 0:
            #     cp_current = np.repeat(max(0, T_current - 200), N)
            if batch_index != 0:
                T_length = States_updated[:, int(cp_current[0]):, :].shape[1]
            print('T_length', T_length, ', cp_current, ', cp_current)
            States_s = copy(States_updated[:, int(cp_current[0]):, :])
            for i in range(p):
                States_s[:, :, i] = transform(States_s[:, :, i])
            out = mean_detect.fit_tuneK(K_list, States_s, Actions_updated[:, int(cp_current[0]):],
                        example="cdist_sparse", param_grid=param_grid_cp, nfold=5, penalty_function='SCAD',
                        select_param_interval=1, B=B_cp,
                        seed=seed*195 + batch_index, init="changepoints", nthread=nthread,
                        changepoints_init=cp_current,
                        max_iter=1, is_only_cluster=1, is_tune_parallel=1, C=0)
            best_out = out.best_model
            g_index = best_out[1]
            print("cluster index:", g_index)


        ### Change point only method
        if method == "changepoint_only":
            g_index = np.repeat(0, N)
            if batch_index == 0:
                kappa_min = int((T_length - 1 - 17) * 0.4)
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
                            example="cdist_sparse", param_grid=param_grid_cp, nfold=5, penalty_function='SCAD', select_param_interval=1,
                            seed=seed*195 + batch_index, init="clustering", epsilon=epsilon, nthread=nthread,
                            kappa_min=kappa_min, kappa_max=kappa_max, max_iter=1,
                            g_index_init_list=[g_index], C=0, B=B_cp,
                            is_tune_parallel=0, is_cp_parallel=1)
            cp_current += out.best_model[2].flatten()
            print("detected change points in the subset data:", out.best_model[2].flatten())
            print("cluster index:", out.best_model.g_index)

        ### Proposed method
        if method == "proposed":
            # print('1')
            if batch_index == 0:
                cp_current = np.repeat(0, N)
                T_length = T_initial
                kappa_min = int((T_length - 1 - 17) * 0.4)
                kappa_max = min(T_length - 4, int((T_length - 1) * 1.2))
            else:
                T_length = States_updated[:, int(np.max(cp_current)):, :].shape[1] #- 1
                kappa_min = 5
                kappa_max = min(T_length - 3, 30)
            print('T_length', T_length, ', cp_current, ', cp_current)
            States_s = copy(States_updated[:, int(np.max(cp_current)):, :])
            for i in range(3):
                States_s[:, :, i] = transform(States_s[:, :, i])
            epsilon = 1 / T_length
            # print('States_s.shape', States_s.shape)
            # try:
            # init with a fixed change point
            changepoints_init = np.repeat(min(10, Actions_updated.shape[1]-1), N)
            # changepoints_init = changepoints_true
            out = mean_detect.fit_tuneK(K_list, States_s, Actions_updated[:, int(np.max(cp_current)):],
                                example="cdist_sparse", param_grid=param_grid_cp, nfold=5, penalty_function='SCAD',
                                select_param_interval=1, B=B_cp,
                                seed = seed*195 + batch_index, init = "changepoints", changepoints_init=changepoints_init,
                                 epsilon=epsilon, nthread=nthread,
                                 kappa_min = kappa_min, kappa_max = kappa_max, max_iter=max_iter,
                                 init_cluster_range = T_length-1-kappa_min, threshold_type=threshold_type,
                                 is_cp_parallel=1, is_tune_parallel=1)
            best_out = out.best_model
            # changepoints = best_out[2]
            cp_current = np.repeat(int(np.max(cp_current)), N) + best_out[2].flatten()  # change_point_detected['integral_emp']
            print("best_K:", out.best_K)
            print("detected change points in the subset data:", best_out[2].flatten())
            print("cluster index:", best_out.g_index)
            g_index = best_out[1]
            print('system_indicator', system_indicator)
            # with open('interdata_batch' + str(batch_index) + '.dat', "wb") as f:
            #     pickle.dump({'States': States_s,
            #                  'Actions': Actions_updated[:, int(np.max(cp_current)):]}, f)
            # except:
            #     print('!!! BUG in mean_detect')
            #     with open('bugdata.dat', "wb") as f:
            #         pickle.dump({'States': States_s,
            #                      'Actions': Actions_updated[:, int(np.max(cp_current)):]}, f)

        print('Finished change point detection. Time: ', datetime.now() - cpTime)


        # %% estimate the optimal policy
        q_all_group = [None] * len(set(g_index))
        q_all_fit = [None] * len(set(g_index))
        print('States_updated.shape', States_updated.shape)
        policyTime = datetime.now()
        for g in np.unique(g_index):
            print('g =', g, ', batch_index =', batch_index)
            # if batch_index % 2 == 0:
            # print('States_updated[g_index == ', g, ', ', cp_current[np.where(g_index == g)[0][0]], ':, :]',
            #       States_updated[g_index == g, cp_current[np.where(g_index == g)[0][0]]:, :].shape)
            # try:
            if sum(g_index == g) == 1:
                q_all_group[g] = None
            else:
                # skip if all actions are the same
                most_recent_cp = cp_current[np.where(g_index == g)[0][0]]
                n_actions = np.sum(Actions_updated[g_index == g, most_recent_cp:].reshape(
                                    (np.sum(g_index == g), -1)))
                dims = np.prod(Actions_updated[g_index == g, most_recent_cp:].reshape(
                                    (np.sum(g_index == g), -1)).shape)
                if n_actions < dims * 0.95 and n_actions > dims * 0.05:
                    print("select_model_cv")
                    if sum(g_index == g) >= 10:
                        out = select_model_cv(
                            States_updated[g_index == g, most_recent_cp:, :].reshape(
                                (np.sum(g_index == g), -1, p)),
                            Rewards_updated[g_index == g, most_recent_cp:].reshape(
                                (np.sum(g_index == g), -1)),
                            Actions_updated[g_index == g, most_recent_cp:].reshape(
                                (np.sum(g_index == g), -1)), param_grid, bandwidth=rbf_bw,
                            qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                            nfold=5, num_threads=nthread, metric=metric)
                        model = out['best_model']
                    else:
                        model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=60, random_state=seed)

                    print(model)
                    print("q_learning")
                    q_all_group[g] = stat.q_learning(
                        States_updated[g_index == g, cp_current[np.where(g_index == g)[0][0]]:, :],
                        Rewards_updated[g_index == g, cp_current[np.where(g_index == g)[0][0]]:],
                        Actions_updated[g_index == g, cp_current[np.where(g_index == g)[0][0]]:], qmodel, degree, gamma,
                        rbf_dim=degree, rbf_bw=rbf_bw)
                    print("fit")
                    q_all_fit[g] = q_all_group[g].fit(model, max_iter=200, tol=1e-6)
                    print("done", g)
                else:
                    q_all_group[g] = None
            # except:
            #     print('!!! BUG in policy learning')
            #     with open(data_path + '/bugdata_' + type_est + '.dat', "wb") as f:
            #         pickle.dump({'States': States_updated[g_index == g, cp_current[np.where(g_index == g)[0][0]]:,
            #                                :].reshape((np.sum(g_index == g), -1, p)),
            #                      'Actions': Actions_updated[g_index == g,
            #                                 cp_current[np.where(g_index == g)[0][0]]:].reshape(
            #                          (np.sum(g_index == g), -1)),
            #                      'Rewards': Rewards_updated[g_index == g,
            #                                 cp_current[np.where(g_index == g)[0][0]]:].reshape(
            #                          (np.sum(g_index == g), -1))}, f)
            #     exit()
        print('Finished. Time: ', datetime.now() - policyTime)

        # %% now we collect the next batch of data following the estimated policy
        T_current += cp_detect_interval
        print("batch_index =", batch_index, "\ncp_index =", cp_index, "\ncp_current =", cp_current, "\nT_to_simulate =", T_current)
        # if the next change point has not been encountered, just keep the current dynamics
        # print('change_points_subsequent[', cp_index + 1, ']', change_points_subsequent[cp_index + 1])
        if T_current <= change_points_subsequent[cp_index + 1]:
            change_or_not = [0, 0, 0]  # whether encountered a change point. change_or_not[0] is the indicator for true group 0
        # if the next change point is encountered, need to make it the change point
        else:
            np.random.seed(seed * batch_index + int(States_updated[0,0,0]))
            change_or_not = np.concatenate((np.random.binomial(1, 0.5, 2), np.array([0])))
            for g in range(2):
                # if change, switch the generating system
                if change_or_not[g] > 0.5:
                    system_indicator[g] = np.abs(system_indicator[g] - 1)
            print("change point encountered", change_or_not)
            cp_index += 1
        print("change_or_not: ", change_or_not)
        print("system_indicator: ", system_indicator)


        # simulate new data following the estimated policy
        # seed_new = int(np.sqrt(np.random.randint(1e6) + seed_new*np.random.randint(10)))

        # S0 = States_updated[:, -1, :]
        States_new = np.zeros([N, cp_detect_interval+1, p])
        Rewards_new = np.zeros([N, cp_detect_interval])
        Actions_new = np.zeros([N, cp_detect_interval])
        for i in range(N):
            # importlib.reload(sim)
            S0 = States_updated[i, -1, :]
            # S0 = S0.reshape((1, S0.shape[0], -1))
            g = g_index_true[i]
            system_settings = system_settings_list[system_indicator[g]]
            if change_or_not[g] == 0:
                system_settings['changepoints'] = [0]
            else:
                # if batch_index == 0:
                #     system_settings['changepoints'] = [change_points_subsequent[cp_index][g_index_true[i]] - T_current + cp_detect_interval]
                # else:
                system_settings['changepoints'] = [change_points_subsequent[cp_index] - T_current + cp_detect_interval]
            # print("i =", i, ", cp =", system_settings['changepoints'])
            # print("system_settings['changepoints']", system_settings['changepoints'])
            # system_settings['changepoints'] = [change_or_not[g] * (change_points_subsequent[cp_index] - T_current + cp_detect_interval)]
            States0, Rewards0, Actions0 = sim.simulate(system_settings, seed=seed*i + i, S0=S0, optimal_policy_model=q_all_group[g_index[i]], epsilon_greedy = 0.2)
            States_new[i, :, :] = States0[0,:,:]
            Rewards_new[i, :] = Rewards0
            Actions_new[i, :] = Actions0

        States_updated = np.concatenate((States_updated, States_new[:,1:,:]), axis = 1)
        Rewards_updated = np.concatenate((Rewards_updated, Rewards_new), axis = 1)
        Actions_updated = np.concatenate((Actions_updated, Actions_new), axis = 1)
        # print('States_updated.shape', States_updated.shape)
        # print('rewards. ', np.mean(Rewards_updated[:, T_initial:]))
        system_list.append(system_indicator)
        sys.stdout.flush()

    #%% compute values
    values = {}
    # discounted reward
    estimated_value = 0.0
    for t in range(T_initial, Rewards_updated.shape[1]):
        estimated_value += Rewards_updated[:,t] * gamma**(t - T_initial)
    values['discounted_reward'] = np.mean(estimated_value)
    values['average_reward'] = np.mean(Rewards_updated[:, T_initial:])
    values['raw_reward'] = Rewards_updated
    return values, system_list




# %% run the evaluation
# method_list = ['oracle', 'proposed', 'overall', 'random', 'kernel0', 'kernel01', 'kernel02', 'kernel03', 'kernel04']
values, system_list = estimate_value(States, Rewards, Actions, type_est=type_est, param_grid=param_grid_policy, basemodel=basemodel)
print(type_est, "discounted reward:", values['discounted_reward'], "\n")
print(type_est, "average reward:", values['average_reward'], "\n")
with open(data_path + "/value_online_" + type_est + ".dat", "wb") as f:
    pickle.dump({'value': values,
                 'system': system_list}, f)


sys.stdout.flush()
print('Finished. Time: ', datetime.now() - startTime)

sys.stdout.close()
sys.stdout = stdoutOrigin




