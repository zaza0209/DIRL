'''
Simulate nonstationary time series data and apply Q-learning.
Data generation mechanism resembles the IHS data
'''
#!/usr/bin/python
import platform, sys, os, pickle
from copy import copy
plat = platform.platform()
print(plat)
if plat == 'macOS-13.3.1-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private/simulation_real")
    sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL_private")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")
elif plat == 'Linux-4.18.0-372.51.1.el8_6.x86_64-x86_64-with-glibc2.28':  # greatlakes
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")

import numpy as np
from datetime import datetime
import functions.simu_mean_detect as mean_detect
import simulation_real.simulate_data_real as sim
from copy import copy
from copy import deepcopy
from collections import namedtuple
from sklearn.tree import DecisionTreeRegressor
from functions.evaluation_separateA import *
import functions.compute_test_statistics_separateA as stat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.optimizers import SGD
from keras.metrics import MeanSquaredError

# Arguments passed

# argv should be: seed, kappa, degree, num_threads
seed = int(sys.argv[1])
effect_size = str(sys.argv[2])
# gamma = float(sys.argv[5])
type_est = 'dqn'
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




#%% Setup DNN model

def create_design_matrix(States, Actions, type = "segmented", scaling = None):
    '''
    Create feature matrix of state and time. Standardized
    '''
    unique_Actions = np.unique(Actions)
    n_actions = len(unique_Actions)
    Actions_stack = Actions.flatten()
    p = States.shape[2] + 1

    # add time as a feature to states
    time_add = np.tile(np.arange(0, States.shape[1]), (States.shape[0], 1))
    time_add = time_add.reshape(States.shape[0], States.shape[1], 1)
    States_full = np.concatenate((States, time_add), axis=2)
    # standardize the features
    if scaling is None:
        scaling = {}
        for i in range(p):
            scaling['mean'] = np.mean(States_full[:, i])
            scaling['sd'] = np.std(States_full[:, i])
            States_full[:, :, i] = (States_full[:, :, i] - scaling['mean']) / scaling['sd']
    else:
        for i in range(p):
            States_full[:, :, i] = (States_full[:, :, i] - scaling['mean']) / scaling['sd']

    if type == "segmented":
        ## stack the states by person, and time: S11, ..., S1T, S21, ..., S2T
        # the current state
        States_stack = States_full[:, :-1 or None, :].transpose(2, 0, 1).reshape(States_full.shape[2], -1).T
        State_features = [None] * n_actions
        for a in unique_Actions:
            a = int(a)
            State_features[a] = States_stack[np.where(Actions_stack == a)[0], :]

        # the next state
        State_next_features = States_full[:, 1:, :].transpose(2, 0, 1).reshape(States_full.shape[2], -1).T
        return State_features, State_next_features, scaling

    elif type == "all":
        # all states
        State_features = States_full[:, :-1 or None, :].transpose(2, 0, 1).reshape(States_full.shape[2], -1).T
        return State_features


def create_q_model():
    '''
    Setup DNN model
    '''
    # define the keras model
    model = Sequential()
    model.add(Dense(32, activation='relu')) #input_dim = State_features[0].shape[1],
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # compile the model
    model.compile(loss='mean_squared_error', optimizer='adam') #, metrics=MeanSquaredError()
    return model



class dqn_agent():
    def __init__(self, States, Actions, Rewards):
        self.State_features, self.State_next_features, self.scaling = create_design_matrix(States, Actions)
        self.Rewards_vec = Rewards.flatten()
        self.actions_unique = np.unique(Actions)
        self.n_actions = len(self.actions_unique)
        # create a list of indices to indicate which action is taken
        self.action_indices = [np.where(Actions.flatten() == a)[0] for a in range(self.n_actions)]
        self.model = None
        self.q_function_list = [None] * self.n_actions

    def fit(self, function_to_create_model, tol = 0.02, max_iter = 10):
        self.model = function_to_create_model()
        # initialize model weights
        self.model.fit(x=self.State_next_features, y=self.Rewards_vec, verbose=0)
        self.model.predict(self.State_next_features, verbose=0)

        # initialize error and number of iterations
        err = 10.0
        err_old = err
        num_iter = 0

        # create separate q functions for different actions
        for a in self.actions_unique:
            a = int(a)
            self.q_function_list[a] = copy(self.model)

        convergence = True
        errors = []
        predicted_old = [np.zeros(len(self.action_indices[a])) for a in range(self.n_actions)]
        predicted = [np.zeros(len(self.action_indices[a])) for a in range(self.n_actions)]
        while abs(err-err_old)/err_old > tol and num_iter <= max_iter:
            Q_max = np.ones(shape=self.Rewards_vec.shape) * (-999)
            for a in self.actions_unique:
                a = int(a)
                # if self.States0[a] is not None:
                # predict the Q value for the next time and find out the maximum Q values for each episode
                Q_max = np.maximum(self.q_function_list[a].predict(self.State_next_features), Q_max)

            # compute TD target
            td_target = self.Rewards_vec + gamma * Q_max
            # update parameter fit
            err = 0.0
            for a in self.actions_unique:
                a = int(a)
                self.q_function_list[a].fit(self.State_features[a], td_target[self.action_indices[a]], verbose=0)
                predicted[a] = self.q_function_list[a].predict(self.State_features[a], verbose=0).flatten()
                err += np.sum((predicted[a] - predicted_old[a]) ** 2)
            err = np.sqrt(err)
            errors.append(err)
            predicted_old = copy(predicted)
            num_iter += 1
            print("iteration", num_iter, ", dqn error:", err)
        if num_iter > max_iter:
            convergence = False
            print("### DQN model did not converge.")

        # return self.model


    def predict(self, States):
        N = States.shape[0]
        T = States.shape[1] - 1
        Actions0 = np.zeros(shape=(2,), dtype='int32')
        design_matrix0 = create_design_matrix(States, Actions0, type = 'all', scaling=self.scaling)
        # print("design_matrix0=", design_matrix0)
        # print(design_matrix0[0,:].toarray())
        opt_reward = np.ones(shape = (N*T,)) * (-999)
        opt_action = np.zeros(shape = (N*T,), dtype = 'int32')
        for a in self.actions_unique:
            a = int(a)
            q_estimated0_a = self.q_function_list[a].predict(design_matrix0, verbose=0).flatten()
            better_action_indices = np.where(q_estimated0_a > opt_reward)
            opt_action[better_action_indices] = a
            opt_reward = np.maximum(opt_reward, q_estimated0_a)
        optimal = namedtuple("optimal", ["opt_reward", "opt_action"])
        return optimal(opt_reward, opt_action)

# q_learn = dqn_agent(States, Actions, Rewards)
# q_learn.fit(create_q_model, tol = 0.02, max_iter = 10)
# q_learn.predict(States[0:2, 0:2, :])


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


method = 'dqn'
changepoints_true = np.repeat([11,17,0], [N_per_cluster]*3, axis=0)
g_index_true = np.repeat([0,1,2], [N_per_cluster]*3, axis=0)
cp_detect_interval = 10
system_indicator = [0, 1, 2]
seed_new = seed
np.random.seed(seed)
system_settings_list[0]["T"] = cp_detect_interval
system_settings_list[1]["T"] = cp_detect_interval
system_settings_list[2]["T"] = cp_detect_interval
States_updated = copy(States)
Rewards_updated = copy(Rewards)
Actions_updated = copy(Actions)
p = States_updated.shape[2]

# number of data batches
n_batch = int((T_new - T_initial) / cp_detect_interval)
cp_index = 0
cp_current = np.repeat(0, N)
T_current = T_initial
T_length = T_initial
change_or_not = [None] * 2
system_list = []
for batch_index in range(n_batch):  #
    print('======= batch_index ======', batch_index)

    cpTime = datetime.now()
    # %% estimate the optimal policy
    q_learn = dqn_agent(States_updated, Actions_updated, Rewards_updated)
    q_learn.fit(create_q_model, tol=0.02, max_iter=10)

    # %% now we collect the next batch of data following the estimated policy
    T_current += cp_detect_interval
    print("batch_index =", batch_index, "\ncp_index =", cp_index, "\ncp_current =", cp_current[0], "\nT_to_simulate =",
          T_current)
    # if the next change point has not been encountered, just keep the current dynamics
    # print('change_points_subsequent[', cp_index + 1, ']', change_points_subsequent[cp_index + 1])
    if T_current <= change_points_subsequent[cp_index + 1]:
        change_or_not = [0, 0, 0]  # whether encountered a change point. change_or_not[0] is the indicator for true group 0
    # if the next change point is encountered, need to make it the change point
    else:
        np.random.seed(seed * batch_index + int(States_updated[0, 0, 0]))
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
    States_new = np.zeros([N, cp_detect_interval + 1, p])
    Rewards_new = np.zeros([N, cp_detect_interval])
    Actions_new = np.zeros([N, cp_detect_interval])
    for i in range(N):
        S0 = States_updated[i, -1, :]
        g = g_index_true[i]
        system_settings = system_settings_list[system_indicator[g]]
        if change_or_not[g] == 0:
            system_settings['changepoints'] = [0]
        else:
            system_settings['changepoints'] = [change_points_subsequent[cp_index] - T_current + cp_detect_interval]
        # print("i =", i, ", cp =", system_settings['changepoints'])
        # print("system_settings['changepoints']", system_settings['changepoints'])
        # system_settings['changepoints'] = [change_or_not[g] * (change_points_subsequent[cp_index] - T_current + cp_detect_interval)]
        States0, Rewards0, Actions0 = sim.simulate(system_settings, seed=seed * i + i, S0=S0,
                                                   optimal_policy_model=q_learn, epsilon_greedy=0.1)
        States_new[i, :, :] = States0[0, :, :]
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



# %% run the evaluation
# method_list = ['oracle', 'proposed', 'overall', 'random', 'kernel0', 'kernel01', 'kernel02', 'kernel03', 'kernel04']
print(type_est, "discounted reward:", values['discounted_reward'], "\n")
print(type_est, "average reward:", values['average_reward'], "\n")
with open(data_path + "/value_online_" + type_est + ".dat", "wb") as f:
    pickle.dump({'value': values,
                 'system': system_list}, f)

sys.stdout.flush()
print('Finished. Time: ', datetime.now() - startTime)

sys.stdout.close()
sys.stdout = stdoutOrigin