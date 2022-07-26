'''
Simulate non-stationary time series data and apply Q-learning.
estimate the optimal reward after identifying change point
'''
#!/usr/bin/python
import platform, sys, os, pickle, re
plat = platform.platform()
if plat == 'macOS-12.3.1-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL/simulation_real")
    sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL")
elif plat == 'Linux-3.10.0-957.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_real")
    sys.path.append("/home/mengbing/research/RL_nonstationary/code2/")
elif plat == 'Linux-3.10.0-1160.6.1.el7.x86_64-x86_64-with-glibc2.17' or plat == 'Linux-3.10.0-1160.45.1.el7.x86_64-x86_64-with-glibc2.17':  # greatlakes
    os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_real")
    sys.path.append("/home/mengbing/research/RL_nonstationary/code2/")
elif plat == 'Linux-3.10.0-1160.15.2.el7.x86_64-x86_64-with-centos-7.9.2009-Core': #gcp
    os.chdir("/home/limengbinggz_gmail_com/rl_nonstationary/simulation_nonstationary_real")
    sys.path.append("/home/limengbinggz_gmail_com/rl_nonstationary/")

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import simulation_real.simulate_data_real as sim

# # # Get the current working directory
# cwd = os.getcwd()
# # Print the current working directory
# print("Current working directory: {0}".format(plat))
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from functions.evaluation_separateA import *



# import importlib
# importlib.reload(sim)
# importlib.reload(stat)
# importlib.reload(plotting)
N = int(100)
# RBFSampler_random_state = 3

# import importlib
# importlib.reload(sim)
# importlib.reload(mean_detect)
# importlib.reload(stat)
# importlib.reload(plotting)

startTime = datetime.now()
seed = 0
np.random.seed(seed)

# %% simulate data
# terminal timestamp
T = 26
#change point
# change_pt = 16
# number of people
# N = int(3)
num_threads = 5
test_setting = "mean" #"cdist"
sim_setting = "largediff"

test_setting == "cdist"
sim_setting = "sametransition"
if sim_setting == "largediff":
    # transition function coefficients
    transition_coefficients = {'cluster1': {'state': np.array([[10, 0.4, -0.04, 0.1],
                                                               [11, -0.4, 0.05, 0.4],
                                                               [1.2, -0.02, 0.03, 0.8]]),
                                            'action': np.array([[0.6, 0.3, 0, 0],
                                                                [-0.4, 0, 0, 0],
                                                                [-0.5, 0, 0, 0]])},
                               'cluster2': {'state': np.array([[8, 0.4, -0.04, 0.1],
                                                               [9, -0.4, 0.05, 0.4],
                                                               [1.0, -0.02, 0.03, 0.8]]),
                                            'action': np.array([[0.5, 0.3, 0, 0],
                                                                [-0.3, 0, 0, 0],
                                                                [-0.4, 0, 0, 0]])}
                               }
    clusters = np.array([[50, 10],
                         [50, 20]])
elif sim_setting == "sametransition":
    # transition function coefficients
    # transition_coefficients = {'cluster1': {'state': np.array([[10, 0.4, -0.04, 0.1],
    #                                                            [11, -0.4, 0.05, 0.4],
    #                                                            [1.2, -0.02, 0.03, 0.8]]),
    #                                         'action': np.array([[0.6, 0.3, 0, 0],
    #                                                             [-0.4, 0, 0, 0],
    #                                                             [-0.5, 0, 0, 0]])},
    #                            'cluster2': {'state': np.array([[10, 0.4, -0.04, 0.1],
    #                                                            [11, -0.4, 0.05, 0.4],
    #                                                            [1.2, -0.02, 0.03, 0.8]]),
    #                                         'action': np.array([[0.6, 0.3, 0, 0],
    #                                                             [-0.4, 0, 0, 0],
    #                                                             [-0.5, 0, 0, 0]])}
    #                            }
    cluster_settings = {'cluster1': {'n': 50, 'changepoints': [11],
                                     'state': [np.array([[10, 0.4, -0.04, 0.1],
                                                        [11, -0.4, 0.05, 0.4],
                                                        [1.2, -0.02, 0.03, 0.8]])]*2,
                                     'action': [np.array([[0.6, 0.3, 0, 0],
                                                         [-0.4, 0, 0, 0],
                                                         [-0.5, 0, 0, 0]]),
                                                np.array([[0.6, 0.3, 0, 0],
                                                          [-0.4, 0, 0, 0],
                                                          [-0.5, 0, 0, 0]])*-1.0
                                                ]
                                     },
                        'cluster2': {'n': 50, 'changepoints': [9, 17],
                                     'state': [np.array([[10, 0.4, -0.04, 0.1],
                                                        [11, -0.4, 0.05, 0.4],
                                                        [1.2, -0.02, 0.03, 0.8]]),
                                               np.array([[10, 0.3, -0.03, 0.15],
                                                         [11.5, -0.5, 0.02, 0.3],
                                                         [1.2, -0.02, 0.03, 0.8]]),
                                               np.array([[10, 0.4, -0.04, 0.1],
                                                         [11, -0.4, 0.05, 0.4],
                                                         [1.2, -0.02, 0.03, 0.8]]),
                                               ],
                                     'action': [np.array([[0.6, 0.3, 0, 0],
                                                         [-0.4, 0, 0, 0],
                                                         [-0.5, 0, 0, 0]]),
                                                np.array([[0.6, 0.3, 0, 0],
                                                          [-0.4, 0, 0, 0],
                                                          [-0.5, 0, 0, 0]]),
                                                np.array([[0.6, 0.3, 0, 0],
                                                          [-0.4, 0, 0, 0],
                                                          [-0.5, 0, 0, 0]])*-1.0
                                                ]
                                     },
                        'cluster3': {'n': 50, 'changepoints': [26],
                                     'state': [np.array([[10, 0.4, -0.04, 0.1],
                                                        [11, -0.4, 0.05, 0.4],
                                                        [1.2, -0.02, 0.03, 0.8]])]*2,
                                     'action': [np.array([[0.6, 0.3, 0, 0],
                                                         [-0.4, 0, 0, 0],
                                                         [-0.5, 0, 0, 0]])]*2
                                     }
                        }






#%% environment setup
# date = '01202022_T50'
test_setting = "cdist"
# create folder under seed if not existing
path_name = 'data/'
if not os.path.exists(path_name):
    os.makedirs(path_name, exist_ok=True)
path_name += "sim" + sim_setting + "_test" + test_setting + "/"
if not os.path.exists(path_name):
    os.makedirs(path_name, exist_ok=True)
path_name += "evaluation/"
if not os.path.exists(path_name):
    os.makedirs(path_name, exist_ok=True)

type_est = 'oracle'
# os.chdir(data_path)
stdoutOrigin = sys.stdout
sys.stdout = open("log_" + type_est + ".txt", "w")

time_terminal = T
qmodel = 'polynomial'
degree = 1
N_new = 300
# T_new = 200 + time_change_pt_true
num_threads = 1

# number of clusters
K = len(clusters)
cluster_sizes = clusters[:, 0]


#%% estimate the value of the estimated policy assuming stationarity in the entire sequence
# importlib.reload(stat)
rbf_bw = 0.1
metric = 'ls'
gamma = 0.9
plot_value = seed < 3
# DecisionTreeRegressor(max_depth=6, min_samples_leaf=70, random_state=50)
def estimate_value(States, Rewards, Actions, param_grid, basemodel, time_change_pt_true):
    # if N > 50:
    #     num_threads = 1
    # np.random.seed(seed)
    out = select_model_cv(States, Rewards, Actions, param_grid, bandwidth=None,
                        qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                        nfold = 5, num_threads = num_threads, metric = metric)
    model = out['best_model']
    print(model)
    q_all = stat.q_learning(States, Rewards, Actions, qmodel, degree, gamma, rbf_dim=degree, rbf_bw=rbf_bw)
    q_all_fit = q_all.fit(model, max_iter=200, tol = 1e-6)

    if plot_value:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        # plt.subplots_adjust(hspace=5)
        for a in range(2):
            tree.plot_tree(q_all.q_function_list[a], ax=axs[a])
            axs[a].set_title('Action ' + str(2*a-1), loc='left')
        # plt.tight_layout()
        fig.savefig('plot_policy' + type_est + '.pdf', bbox_inches='tight', pad_inches = 0.5)
        plt.close('all')
        # plt.show()

    _, Rewards_new, _ = simulate(N_new, optimal_policy_model=q_all, T0=time_change_pt_true)
    estimated_value = 0.0
    for t in range(T_new-time_change_pt_true):
        estimated_value += Rewards_new[:,t] * gamma**t
    return estimated_value



sim_dat = sim.simulate_data(T, clusters, transition_coefficients)
param_grid = {"max_depth": [3, 5, 6],#
              "min_samples_leaf": [30, 40, 50, 60]}#

nsims = 30
for seed in range(nsims):
    print("seed", seed)
    # simulate data
    States, Rewards, Actions = sim_dat.simulate(seed, burnin=60, T1=T)

    # oracle
    cum_i = 0
    for k in range(K):
        # extract individuals in cluster k
        States_k = States[cum_i:cum_i+cluster_sizes[k], :, :]
        Rewards_k = Rewards[cum_i:cum_i+cluster_sizes[k], :]
        Actions_k = Actions[cum_i:cum_i+cluster_sizes[k], :]












#%% generate data for estimating the optimal policy
def simulate(N=100, optimal_policy_model = None, T0=0, T1=300):
    sim_dat = sim.simulate_data(N, T, time_change_pt_true)
    States, Rewards, Actions = sim_dat.simulate(seed, burnin = 60, optimal_policy_model=optimal_policy_model, T0=T0, T1=T1)
    return States, Rewards, Actions

# importlib.reload(sim)
States, Rewards, Actions = simulate(N, optimal_policy_model = None, T0=0, T1=T)
basemodel = DecisionTreeRegressor(random_state=seed)

#%% estimate the value of the estimated policy assuming stationarity in the entire sequence
# importlib.reload(stat)
rbf_bw = 0.1
metric = 'ls'
# model = DecisionTreeRegressor(random_state=seed, max_depth=3, min_samples_leaf=80)
# DecisionTreeRegressor(max_depth=6, min_samples_leaf=70, random_state=50)
def estimate_value(States, Rewards, Actions, param_grid, basemodel):
    # if N > 50:
    #     num_threads = 1
    # np.random.seed(seed)
    out = select_model_cv(States, Rewards, Actions, param_grid, bandwidth=None,
                        qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                        nfold = 5, num_threads = num_threads, metric = metric)
    model = out['best_model']
    print(model)
    q_all = stat.q_learning(States, Rewards, Actions, qmodel, degree, gamma, rbf_dim=degree, rbf_bw=rbf_bw)
    q_all_fit = q_all.fit(model, max_iter=200, tol = 1e-6)

    if plot_value:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        # plt.subplots_adjust(hspace=5)
        for a in range(2):
            tree.plot_tree(q_all.q_function_list[a], ax=axs[a])
            axs[a].set_title('Action ' + str(2*a-1), loc='left')
        # plt.tight_layout()
        fig.savefig('plot_policy' + type_est + '.pdf', bbox_inches='tight', pad_inches = 0.5)
        plt.close('all')
        # plt.show()

    _, Rewards_new, _ = simulate(N_new, optimal_policy_model=q_all, T0=time_change_pt_true)
    estimated_value = 0.0
    for t in range(T_new-time_change_pt_true):
        estimated_value += Rewards_new[:,t] * gamma**t
    return estimated_value


param_grid = {"max_depth": [3, 5, 6],#
              "min_samples_leaf": [30, 40, 50, 60]}#
# param_grid = {"max_depth": [3, 5, 6, 8],#
#               "min_samples_leaf": [50, 60, 70, 80]}#
# param_grid = {"max_depth": [3],#
#               "min_samples_leaf": [50,60,80]}#


# fig = plt.figure(figsize=(25, 10))
# _ = tree.plot_tree(q_all.q_function_list[1])
# plt.show()
# # fig.savefig("decistion_tree_estimated.png")
#
# class always_pick_action_0():
#     def __init__(self):
#         self.x = 0
#
#     def predict(self, x):
#         # return np.zeros(shape=(x.shape[0],1), dtype = 'int32')
#         return np.ones(shape=(x.shape[0],1), dtype = 'int32')
#
# time_change_pt = 80
# States2 = copy(States)
# Rewards2 = copy(Rewards)
# Actions2 = copy(Actions)
# States = States[:, time_change_pt:, :]
# Rewards = Rewards[:, time_change_pt:]
# Actions = Actions[:, time_change_pt:]
# #
# pick_0 = always_pick_action_0()
# q_all.model = pick_0
# _, Rewards_new, Actions_new = simulate(N_new, optimal_policy_model=1, T0=time_change_pt_true)
# estimated_value = 0.0
# for t in range(T - time_change_pt_true):
#     estimated_value += Rewards_new[:, t] * gamma ** t
# print(np.mean(estimated_value))

# import matplotlib.patches as mpatches
# idx = 0
# a0 = mpatches.Patch(color='r', label='Action = -1')
# a1 = mpatches.Patch(color='g', label='Action = 1')
# colors = ['r', 'g']
# x = np.arange(Rewards.shape[1])
# y = Rewards[idx,:]
# for x1, x2, y1,y2 in zip(x, x[1:], y, y[1:]):
#     # print(y1)
#     plt.plot([x1, x2], [y1, y2], c=colors[Actions[idx, int(x1)]])
# plt.legend(handles=[a0, a1])
# plt.show()



#%% estimate the value of the observed policy
if type_est == 'overall':
    estimated_value = 0.0
    for t in range(T):
        estimated_value += Rewards[:,t] * gamma**t
    pickle.dump(np.mean(estimated_value), open("value_observed_gamma" + re.sub("\.", "", str(gamma)) + ".dat", "wb"))
    print("Observed reward:", np.mean(estimated_value), "\n")
    sys.stdout.flush()



#%% overall policy: assume stationarity throughout
if type_est == 'overall':
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_overall = estimate_value(States, Rewards, Actions, param_grid, basemodel=model)
    estimated_value_overall = np.mean(estimated_value_overall)
    print("Overall estimated reward:", estimated_value_overall, "\n")
    pickle.dump(estimated_value_overall, open("value_overall_gamma" + re.sub("\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()
    if plot_value:
        fig = plt.hist(estimated_value_overall, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of overall values')
        plt.savefig("hist_value_overall_gamma" + re.sub("\.", "", str(gamma)) + ".png")



# %% estimate the oracle policy: piecewise Q function before and after change point
#%% fit the Q model with known change point
if type_est == 'oracle':
    time_change_pt = time_change_pt_true
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle = estimate_value(States[:, time_change_pt:, :], Rewards[:, time_change_pt:], Actions[:, time_change_pt:],
                                            param_grid, model)
    estimated_value_oracle = np.mean(estimated_value_oracle)
    print("Oracle estimated reward:", estimated_value_oracle, "\n")
    pickle.dump(estimated_value_oracle, open("value_oracle_gamma" + re.sub("\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()
    if plot_value:
        fig = plt.hist(estimated_value_oracle, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\.", "", str(gamma)) + ".png")


# %% estimate the oracle policy at time 80: piecewise Q function before and after change point
# %% fit the Q model with known change point
if type_est == 'oracle70':
    time_change_pt = int(70)
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle70 = np.mean(
        estimate_value(States[:, time_change_pt:, :], Rewards[:, time_change_pt:], Actions[:, time_change_pt:],
                       param_grid, model))
    print("Oracle 70 estimated reward:", estimated_value_oracle70, "\n")
    pickle.dump(estimated_value_oracle70, open("value_oracle70_gamma" + re.sub("\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()

#%% fit the Q model with known change point
if type_est == 'oracle80':
    time_change_pt = int(80)
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle80 = np.mean(estimate_value(States[:, time_change_pt:, :], Rewards[:, time_change_pt:], Actions[:, time_change_pt:], param_grid, model))
    print("Oracle 80 estimated reward:", estimated_value_oracle80, "\n")
    pickle.dump(estimated_value_oracle80, open("value_oracle80_gamma" + re.sub("\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()


# %% estimate the oracle policy at time 60: piecewise Q function before and after change point
#%% fit the Q model with known change point
if type_est == 'oracle60':
    time_change_pt = int(60)
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle60 = np.mean(estimate_value(States[:, time_change_pt:, :], Rewards[:, time_change_pt:], Actions[:, time_change_pt:], param_grid, model))
    pickle.dump(estimated_value_oracle60, open("value_oracle60_gamma" + re.sub("\.", "", str(gamma)) + ".dat", "wb"))
    print("Oracle 60 estimated reward:", estimated_value_oracle60, "\n")
    sys.stdout.flush()



# %% estimate the proposed policy: piecewise Q function before and after the estimated change point
#%% fit the Q model with known change point
if type_est == 'proposed':
    method = "sequential"
    time_change_pt_seq = int(pickle.load(open('changept_' + method + '.dat', "rb")))
    print("sequential:", time_change_pt_seq)
    time_change_pt = time_change_pt_seq
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_seq = estimate_value(States[:, time_change_pt:, :], Rewards[:, time_change_pt:], Actions[:, time_change_pt:], param_grid, model)
    estimated_value_seq = np.mean(estimated_value_seq)
    print("Sequential estimated reward:", estimated_value_seq)
    pickle.dump(estimated_value_seq, open("value_" + method + "_gamma" + re.sub("\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()



# %% estimate the proposed policy: piecewise Q function before and after the estimated change point
#%% fit the Q model with known change point
if type_est == 'random':
    method = "random"
    time_change_pt_random = np.random.randint(0, T-1)# sample(range(1,T-1), 1)[0]
    print("\nrandom:", time_change_pt_random)
    time_change_pt = time_change_pt_random
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_random = estimate_value(States[:, time_change_pt:, :], Rewards[:, time_change_pt:], Actions[:, time_change_pt:], param_grid, model)
    estimated_value_random = np.mean(estimated_value_random)
    print("Random estimated reward:", estimated_value_random, "\n")
    pickle.dump(estimated_value_random, open("value_" + method + "_gamma" + re.sub("\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()



# %% estimate the baseline optimal rewards using kernel-based FQI
# States2 = copy(States)
# Rewards2 = copy(Rewards)
# Actions2 = copy(Actions)
# States = States[:, time_change_pt:, :]
# Rewards = Rewards[:, time_change_pt:]
# Actions = Actions[:, time_change_pt:]
# States = copy(States2)
# Rewards = copy(Rewards2)
# Actions = copy(Actions2)

if type_est == 'kernel0':
    param_grid = {"max_depth": [3, 5, 6],  #
                  "min_samples_leaf": [10, 20, 40, 50]}  #
    method = "kernel"
    time_change_pt = T - 1
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_kernel0 = estimate_value(States[:, time_change_pt:, :], Rewards[:, time_change_pt:],
                                            Actions[:, time_change_pt:], param_grid, model)
    estimated_value_kernel0 = np.mean(estimated_value_kernel0)
    print("kernel0 estimated reward:", estimated_value_kernel0, "\n")
    pickle.dump(estimated_value_kernel0, open("value_" + method + "_bandwidth0" + ".dat", "wb"))
    sys.stdout.flush()


if ''.join([i for i in type_est if not i.isdigit()]) == 'kernel':
    bandwidth = float(''.join([i for i in type_est if i.isdigit()])) * 0.1
    method = 'kernel'
    num_threads = 1
    param_grid = {"max_depth": [3, 5, 6],#
                  "min_samples_leaf": [30, 40, 50, 60]}#
    # param_grid = {"max_depth": [3, 5, 6, 8],#
    #               "min_samples_leaf": [60, 70, 80, 90]}#
    def gaussian_rbf(x, bandwidth):
        return np.exp(- (x / bandwidth)**2 )

    def FQI_kernel(bandwidth, seed = 50):

        sampling_probs = [gaussian_rbf((T-t) / T, bandwidth) for t in range(T)]
        sampled_time_points = np.random.choice(range(T), num_sampled_times + 1, p=sampling_probs / sum(sampling_probs))
        model = DecisionTreeRegressor(random_state=seed)

        out = select_model_cv(States, Rewards, Actions, param_grid, rbf_bw, qmodel='polynomial', gamma=gamma,
                              model=model, max_iter=200, tol=1e-4, nfold = 5, num_threads = num_threads,
                              sampled_time_points = sampled_time_points, kernel_regression=True, metric=metric)
        model = out['best_model']
        print(model)

        # obtain (St, At, Rt)
        # Q learning after change point
        q_all = stat.q_learning(States[:,sampled_time_points,:], Rewards[:,sampled_time_points[:-1]], Actions[:,sampled_time_points[:-1]],
                                qmodel, degree, gamma)
        # obtain (St+1)
        q_all.States1 = q_all.create_design_matrix(States = States[:,sampled_time_points+1,:], Actions= np.zeros((N, num_sampled_times), dtype='int32'), type='current')
        q_all_fit = q_all.fit(model, max_iter=300, tol = 1e-6)

        # simulate new data
        _, Rewards_new, Actions_new = simulate(N_new, optimal_policy_model=q_all, T0=time_change_pt_true)
        estimated_value = 0.0
        for t in range(T-time_change_pt_true):
            estimated_value += Rewards_new[:,t] * gamma**t
        estimated_value = np.mean(estimated_value)

        print("bandwidth =", bandwidth)
        print("Kernel estimated reward:", estimated_value, "\n")
        pickle.dump(estimated_value, open("value_" + method + "_bandwidth" + re.sub("\.", "", str(bandwidth)) + ".dat", "wb"))
        return 0

    # run kernel method
    num_sampled_times = 200
    if bandwidth > 0:
        FQI_kernel(bandwidth=round(bandwidth,1))
        sys.stdout.flush()


print('Finished. Time: ', datetime.now() - startTime)

sys.stdout.close()
sys.stdout = stdoutOrigin