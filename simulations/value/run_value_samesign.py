# -*- coding:utf-8 -*-
'''
Simulate non-stationary time series data and apply Q-learning.
estimate the optimal reward after identifying change point
'''
#!/usr/bin/python
import platform, sys, os, re, pickle
from copy import deepcopy
from copy import copy
plat = platform.platform()
print(plat)
if plat == 'Windows-10-10.0.14393-SP0': ##local
    os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu/tuneK_iterations/value")
    sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
elif plat == 'Linux-5.10.0-18-cloud-amd64-x86_64-with-glibc2.31' or plat == 'Linux-3.10.0-1160.53.1.el7.x86_64-x86_64-with-centos-7.6.1810-Core':  # biostat cluster
    os.chdir("/home/xx/heterRL/tuneK_iterations/value_samesign")
    sys.path.append("/home/xx/heterRL")
else:
    os.chdir("/home/xx/heterRL/tuneK_iterations/value_samesign")
    sys.path.append("/home/xx/heterRL")

from sklearn import tree
import numpy as np
from datetime import datetime
# import test_stat.simulate_data_1d as sim
from functions.simulate_data_1d_flexible import *
from sklearn.tree import DecisionTreeRegressor
import functions.compute_test_statistics_separateA as stat
from functions.evaluation import *
import functions.simu_mean_detect as mean_detect
# parallel jobs
from joblib import Parallel, delayed
# Arguments passed
#%%
# argv should be: seed, kappa, degree, num_threads
seed = int(sys.argv[1])
type_est = str(sys.argv[2])
N = int(sys.argv[3])
T_new = int(sys.argv[4])
trans_setting = sys.argv[5]
nthread = int(sys.argv[6])
cov = float(sys.argv[7])
cp_detect_interval = int(sys.argv[8])
is_tune_parallel = int(sys.argv[9])
is_cp_parallel= int(sys.argv[10])
effect_size = sys.argv[11]
print('type_est', type_est)
print('trans_setting', trans_setting)
# seed = 10
# N=100
# trans_setting = 'pwconst2'
reward_setting = 'homo'
gamma = 0.9
# type_est = "proposed"


startTime = datetime.now()

np.random.seed(seed)
plot_value = seed < 5

# %% simulate data
# N = 200
# terminal timestamp
T_initial = 50 #100
# delta=0.5
# gamma=0.95
# dimension of X0
p = 1
K=2

max_iter = 10
K_list = list(range(1,5))
# mean vector of X0
mean0 = 0
# diagonal covariance of X0
cov0 = 0.5

# mean vector of random errors zt
mean = 0
# diagonal covariance of random errors zt


#%% environment for saving online values
# method = ''.join([i for i in type_est if not i.isdigit()])
# append_name =  "Tnew"+str(T_new)+ "_cov"+str(cov)
if effect_size == "strong":
    effect_size_factor = 0.40
elif effect_size == "moderate":
    effect_size_factor = 0.35
elif effect_size == "weak":
    effect_size_factor = 0.25
else:
    effect_size_factor = int(effect_size)

if not os.path.exists('results'):
    print('not results')
    os.makedirs('results', exist_ok=True)
print(os.getcwd())
# data_path = 'results/' + 'trans' + trans_setting +'_gamma' + \
#             re.sub("\\.", "", str(gamma)) +'/N' + str(N) +'/'+ type_est + append_name+\
#                 '/seed_'+str(seed)
data_path = 'results/trans' + trans_setting +'/effect_size_samesign'+str(effect_size_factor)+'/N' + str(N) +'/Tnew_' +\
            str(T_new)+'_type_'+type_est+'_cpitv'+ str(cp_detect_interval)+'/cov'+str(cov) + '/seed'+str(seed) 
if not os.path.exists(data_path):
    print('data_path', os.path.exists(data_path))
    os.makedirs(data_path, exist_ok=True)

print(data_path)
os.chdir(data_path)
file_name = 'seed_'+str(seed)+'.dat'
if os.path.exists(file_name):
    exit()
    
stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "w")
print("\nName of Python script:", sys.argv)
print(data_path)
sys.stdout.flush()

qmodel = 'polynomial'
degree = 1

num_threads = 1
#%% generate data for estimating the optimal policy

def transition_function1(St, At, t, mean=0, cov=cov):
    '''
    Generate time-homogeneous transition at time t
    :param St: State at time t
    :param At: action in {0,1} at time t
    :param t: time t
    :return: a scalar of state
    '''
    # noise = np.random.normal(mean, cov, 1)[0] #len(At)
    # new_St = 0.5 * (2.0 * At - 1.0) * St + noise
    # print("noise =", noise)
    # print("new_St =", new_St)
    return (0.5 -effect_size_factor)* (2.0 * At - 1.0) * St

def transition_function2(St, At, t, mean=0, cov=cov):
    '''
    Generate time-homogeneous transition at time t
    # :param St: State at time t
    # :param At: action in {0,1} at time t
    # :param t: time t
    :return: a scalar of state
    '''
    return (0.5+effect_size_factor) * St * (2.0 * At - 1.0)

def reward_function_homo(St, At, t):
    '''
    Generate time-homogeneous reward function in time consisting of 2 pieces
    t = the time at which we generate a reward
    :return: a scalar of reward at time t
    '''
    return 0.25*St**2 * (2.0 * At - 1.0) * effect_size_factor + 4*St

system_settings_list = [{'N': 1, 'T': T_initial-1,
                   'changepoints': [50],
                   'state_functions': [transition_function1, transition_function2],
                   'reward_functions': [reward_function_homo, reward_function_homo],
                   'state_change_type': trans_setting,#,
                   'reward_change_type': 'homogeneous',
                   'delta':0.1
                    },
                   {'N': 1, 'T': T_initial-1,
                                      'changepoints': [50],
                                      'state_functions': [transition_function2, transition_function1],
                                      'reward_functions': [reward_function_homo, reward_function_homo],
                                      'state_change_type': trans_setting,#,
                                      'reward_change_type': 'homogeneous',
                                      'delta':0.1
                                       }]


States = np.zeros([N, T_initial, p])
Rewards = np.zeros([N, T_initial-1])
Actions = np.zeros([N, T_initial-1])
changepoints_true = np.repeat(0, N)
g_index_true = np.repeat([0,1], int(N/2))
system_indicator = [0,1]
if trans_setting == "smooth":
    changepoint_list = [int(T_initial/2)  + int(0.2 * T_initial)- 1+2, int(T_initial/2) - int(0.2 * T_initial) - 1+2] #        ?29     ?  30     ?    μ transition function     
else:
    changepoint_list = [int(T_initial/2)  + int(0.2 * T_initial)- 1, int(T_initial/2) - int(0.2 * T_initial) - 1] #        ?29     ?  30     ?    μ transition function     

for i in range(N):
    if i < int(N/2):
        # print('1')
        changepoint = changepoint_list[0]
        system_settings = system_settings_list[0]
        system_settings['changepoints'] = [changepoint_list[0]]
    else:
        # print('2')
        changepoint = changepoint_list[1]
        system_settings = system_settings_list[1]
        system_settings['changepoints'] = [changepoint_list[1]]
    States0, Rewards0, Actions0 = simulate(system_settings, seed = seed*i+i, cov=cov)
    States[i, :, :] = States0[:, :, :]
    Rewards[i, :] = Rewards0
    Actions[i, :] = Actions0
    changepoints_true[i, ] = int(changepoint)
print('finish data generating')

def transform(x):
    return (x - np.mean(x)) / np.std(x)
# # States = copy(States)
# for i in range(1):
#     States[:,:,i] = transform(States[:,:,i])

#%% estimate the value of the estimated policy
rbf_bw = 0.1
metric = 'kerneldist'
param_grid = {"max_depth": [3, 5],#
              "min_samples_leaf": [50, 60, 70]}#
basemodel = DecisionTreeRegressor(random_state=seed)
# model=DecisionTreeRegressor(random_state=10, max_depth=3, min_samples_leaf = 60)
N_new = N
# T_new = 250
# create a random list of change points
change_points_subsequent = [changepoint_list, T_initial + np.random.poisson(40)]
change_point_next = change_points_subsequent[1]
while change_point_next < T_new - 30:
    change_point_next += np.random.poisson(40)
    change_points_subsequent.append(change_point_next)
change_points_subsequent = change_points_subsequent[:-1]
print("change_points =", change_points_subsequent)
change_points_subsequent.append(T_new)

system_list = []
# model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=50, random_state=480)
def estimate_value(States, Rewards, Actions, type_est, param_grid, basemodel):
    method = ''.join([i for i in type_est if not i.isdigit()])
    print('method', method)
    # cp_detect_interval = cp_detect_interval
    seed_new = seed
    np.random.seed(seed)
    system_settings_list[0]["T"] = cp_detect_interval
    system_settings_list[1]["T"] = cp_detect_interval
    
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
    change_or_not = [None]*2
    for batch_index in range(n_batch):
        print('======= batch_index ======', batch_index)
        #%% first, detect change point
        # cp_current = change_points_subsequent[cp_index]
        cpTime = datetime.now()
        if method == "overall":
            # if T_current > 200 and T_current % 100 == 0:
            #     cp_current = np.repeat(max(0, T_current - 200), N)
            print("cp_current =", cp_current)
            g_index = np.repeat(0, N)
        if method == "oracle":
            if batch_index == 0:
                cp_current = changepoints_true# np.minimum(changepoints_true, np.repeat(T_current, N))
                g_index = g_index_true
            else:
                # if batch_index == 1:
                #     cp_current = np.maximum(cp_current, changepoints_true)
                # else:
                for g in range(2):
                    if change_or_not[g] != 0:
                        cp_current[g_index_true == g] = np.repeat(change_points_subsequent[cp_index], int(N/2))
                if system_indicator[0] == system_indicator[1]:
                    g_index = np.repeat(0, N)
                else:
                    g_index = g_index_true
            # if batch_index == 0:
            #     g_index = g_index_true
            # else:
            #     if system_indicator[0] == system_indicator[1]:
            #         g_index = np.repeat(0, N)
            #     else:
            #         g_index = g_index_true
        if method == "only_clusters":
            # if T_current > 200 and T_current % 100 == 0:
            #     cp_current = np.repeat(max(0, T_current - 200), N)
            if batch_index != 0:
                T_length = States_updated[:, int(cp_current[0]):, :].shape[1] 
            print('T_length', T_length, ', cp_current, ',cp_current)
            States_s = copy(States_updated[:, int(cp_current[0]):, :])
            for i in range(p):
                States_s[:, :, i] = transform(States_s[:, :, i])
            # out = mean_detect.fit_tuneK(K_list, States_s, Actions_updated[:,  int(cp_current[0]):],
            #                      seed = seed+batch_index, init = "changepoints", nthread=nthread,changepoints_init =cp_current, 
            #                      max_iter=1,is_only_cluster=1, is_tune_parallel=is_tune_parallel)
            # best_out = out.best_model
            # g_index = best_out[1]
            out = mean_detect.fit(States_s, Actions_updated[:,  int(cp_current[0]):],
                        seed = seed+batch_index, init = "changepoints", nthread=nthread, changepoints_init =cp_current,
                        max_iter=1,is_only_cluster=1, C=0, K=K_list)
            g_index = out[1]

        if method == "only_cp":
            g_index = np.repeat(0, N)
            if batch_index == 0:
                kappa_min = int((T_initial- 1 - np.max(changepoints_true))*0.8)
                kappa_max = min(T_initial-1, int((T_initial- 1 - np.min(changepoints_true))*1.2))
            else:
                T_length = States_updated[:, int(np.max(cp_current)):, :].shape[1] #- 1
                kappa_min = 5
                kappa_max = min(T_length - 10, 40)
            States_s = copy(States_updated[:, int(np.max(cp_current)):, :])
            for i in range(p):
                States_s[:, :, i] = transform(States_s[:, :, i])
            epsilon = 1/T_length
            print('!!!! is_cp_parallel', is_cp_parallel)
            out = mean_detect.fit_tuneK([1], States_s, Actions_updated[:, int(np.max(cp_current)):],
                                 seed = seed+batch_index, init = "clustering", epsilon=epsilon, nthread=nthread,
                                 kappa_min = kappa_min, kappa_max = kappa_max, max_iter=1, 
                                 g_index_init_list=[g_index],
                                 is_tune_parallel=0, is_cp_parallel = is_cp_parallel)
            cp_current += out.best_model[2]
        if method == "proposed":
            # print('1')
            if batch_index == 0:
                cp_current = np.repeat(0,N)
                # kappa_min = 10 
                # kappa_max = 49
                kappa_min = int((T_initial- 1 - np.max(changepoints_true))*0.8)
                kappa_max = min(T_initial-1, int((T_initial- 1 - np.min(changepoints_true))*1.2))
            else:
                T_length = States_updated[:, int(np.max(cp_current)):, :].shape[1] #- 1
                kappa_min = 5
                kappa_max = min(T_length - 10, 40)
            print('T_length', T_length, ', cp_current, ',cp_current)
            States_s = copy(States_updated[:, int(np.max(cp_current)):, :])
            for i in range(1):
                States_s[:, :, i] = transform(States_s[:, :, i])
            epsilon = 1/T_length
            print('States_s.shape', States_s.shape)
            try:
                out = mean_detect.fit(States_s, Actions_updated[:, int(np.max(cp_current)):],
                    seed = seed, init = "clustering", epsilon=epsilon,  nthread=nthread,
                    kappa_min = kappa_min, kappa_max = kappa_max, max_iter=max_iter, 
                    K=K_list, init_cluster_range = T_length-1-kappa_min,
                    is_cp_parallel=is_cp_parallel, C=2)
                cp_current = np.repeat(int(np.max(cp_current)), N)+ out.changepoints #change_point_detected['integral_emp']
                g_index = out.g_index
                print('system_indicator',system_indicator)
            except:
                print('!!! BUG in mean_detect')
                with open('bugdata.dat' , "wb") as f:
                    pickle.dump({'States':States_s,
                                  'Actions':Actions_updated[:, int(np.max(cp_current)):]}, f)
      
        print('Finished. Time: ', datetime.now() - cpTime)
        #%% estimate the optimal policy
        q_all_group = [None]*len(g_index)
        q_all_fit = [None]*len(g_index)
        print('States_updated.shape', States_updated.shape)
        policyTime = datetime.now()
        for g in np.unique(g_index):
            print('g', g, ', batch_index, ',batch_index)
            # if batch_index % 2 == 0:
            print('States_updated[g_index == ',g,', ', cp_current[np.where(g_index==g)[0][0]], ':, :]',States_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:, :].shape)
            try:
                out = select_model_cv(States_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:, :].reshape((np.sum(g_index == g), -1, p)), 
                                      Rewards_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:].reshape((np.sum(g_index == g), -1)), Actions_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:].reshape((np.sum(g_index == g), -1)), param_grid, bandwidth=rbf_bw,
                                      qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                                      nfold = 5, num_threads = nthread, metric = metric)
                model = out['best_model']
                print(model)
                q_all_group[g] =stat.q_learning(States_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:, :], Rewards_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:], Actions_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:], qmodel, degree, gamma, rbf_dim=degree, rbf_bw=rbf_bw)
                q_all_fit[g] = q_all_group[g].fit(model, max_iter=200, tol = 1e-6)
            except:
                print('!!! BUG in policy learning')
                with open('bugdata.dat' , "wb") as f:
                    pickle.dump({'States':States_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:, :].reshape((np.sum(g_index == g), -1, p)),
                                 'Actions':Actions_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:].reshape((np.sum(g_index == g), -1)),
                                 'Rewards':Rewards_updated[g_index == g, cp_current[np.where(g_index==g)[0][0]]:].reshape((np.sum(g_index == g), -1))}, f)
                exit()
        print('Finished. Time: ', datetime.now() - policyTime)
        #%% now we collect the next batch of data following the estimated policy
        T_current += cp_detect_interval
        print("batch_index =", batch_index, "cp_index =", cp_index, "cp_current =", cp_current, "T_to_simulate =", T_current)
        # if the next change point has not been encountered, just keep the current dynamics
        print('change_points_subsequent[',cp_index+1,']', change_points_subsequent[cp_index+1])
        if T_current -1 <= change_points_subsequent[cp_index+1]:
            change_or_not = [0,0] # whether encountered a change point. change_or_not[0] is the indicator for true group 0 
        # if the next change point is encountered, need to make it the change point
        else:
            np.random.seed(seed*batch_index)
            change_or_not = np.random.binomial(1, 0.5, 2)
            for g in range(2):
                # if change, switch the generating system
                if change_or_not[g] > 0.5:
                    system_indicator[g] = np.abs(system_indicator[g]-1)                    
            print("change print encountered", change_or_not)
            cp_index += 1
        # print(system_settings_batch['changepoints'])

        # simulate new data following the estimated policy
        seed_new = int(np.sqrt(np.random.randint(1e6) + seed_new*np.random.randint(10)))

        # S0 = States_updated[:, -1, :]
        States_new = np.zeros([N, cp_detect_interval+1, p])
        Rewards_new = np.zeros([N, cp_detect_interval])
        Actions_new = np.zeros([N, cp_detect_interval])
        for i in range(N):
            S0 = States_updated[i, -1, :]
            S0 = S0.reshape((1, S0.shape[0], -1))
            g = g_index_true[i]
            system_settings = system_settings_list[system_indicator[g]]
            if change_or_not[g] == 0:
                system_settings['changepoints'] = [0]
            else:
                # if batch_index == 0: 
                #     system_settings['changepoints'] = [change_points_subsequent[cp_index][g_index_true[i]] - T_current + cp_detect_interval]
                # else:
                system_settings['changepoints'] = [change_points_subsequent[cp_index] - T_current + cp_detect_interval]
            # print("system_settings['changepoints']", system_settings['changepoints'])
            # system_settings['changepoints'] = [change_or_not[g] * (change_points_subsequent[cp_index] - T_current + cp_detect_interval)]
            States0, Rewards0, Actions0 = simulate(system_settings,seed=seed_new*i + i, S0=S0, optimal_policy_model=q_all_group[g_index[i]], cov=cov)
            States_new[i, :, :] = States0[0,:,:]
            Rewards_new[i, :] = Rewards0
            Actions_new[i, :] = Actions0

        States_updated = np.concatenate((States_updated, States_new[:,1:,:]), axis = 1)
        Rewards_updated = np.concatenate((Rewards_updated, Rewards_new), axis = 1)
        Actions_updated = np.concatenate((Actions_updated, Actions_new), axis = 1)
        print('States_updated.shape', States_updated.shape)
        print('rewards. ', np.mean(Rewards_updated[:, T_initial:]))
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
# if type_est != 'behavior':
#     method_list = ['oracle', 'proposed', 'overall', 'random', 'kernel0', 'kernel01', 'kernel02', 'kernel03', 'kernel04']
value, system_list = estimate_value(States, Rewards, Actions, type_est=type_est, param_grid=param_grid, basemodel=basemodel)
print('Finished. Time: ', datetime.now() - startTime)
print(type_est, "discounted reward:", value['discounted_reward'], "\n")
print(type_est, "average reward:", value['average_reward'], "\n")
with open(file_name , "wb") as f:
    pickle.dump({'value':value,
                 'system':system_list}, f)

# else:
#     system_settings['T'] = 200
#     States, Rewards, Actions = simulate(system_settings, seed=seed)
#     #%% compute values
#     values = {}
#     # discounted reward
#     estimated_value = 0.0
#     for t in range(T_initial, Rewards.shape[1]):
#         estimated_value += Rewards[:,t] * gamma**t
#     values['discounted_reward'] = np.mean(estimated_value)
#     values['discounted_reward'] = np.mean(Rewards[:, T_initial:])
#     values['raw_reward'] = Rewards_updated



sys.stdout.flush()


sys.stdout.close()
sys.stdout = stdoutOrigin
