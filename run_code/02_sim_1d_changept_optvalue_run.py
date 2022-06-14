# '''
# Estimate the optimal reward after identifying change point.
# The policies evaluated include:
# proposed: using data on [T - kappa^*, T], where kappa^* is the change point detected by isotonic regression
# overall: using data on [0, T]
# behavior: the behavioral policy with A_t \in {-1, 1} and P(A_t = 1) = P(A_t = -1) = 0.5
# random: pick a random change point kappa^**, and evaluate using data on [T - kappa^**, T]. Repeat the process
#     for 20 times and take the average value
# kernel: kernel regression method to deal with nonstationarity as described in the paper. Multiple bandwidths
#     are taken: 0.2, 0.4, 0.8, and 1.6.
# '''
#!/usr/bin/python
import platform, sys, os, re, pickle
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed
# import simu.compute_test_statistics as stat
# os.chdir("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/simulation_1d")
# sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
import simu.simulate_data_pd as sim
# from functions.evaluation_separateA import *
from simu.ev2 import *
'''
Arguments passed:
- seed: int. random seed to generate data
- trans_setting: string. scenario of the transition function. Takes value from 'homo', 'pwconst2', or 'smooth'
- reward_setting: string. scenario of the reward function. Takes value from 'homo', 'pwconst2', or 'smooth'
- gamma: float. discount factor for the cumulative discounted reward. between 0 and 1
- N: int. number of individuals
- type_est: string. the type of policy to be estimated. Takes values from 'overall', 'oracle', 'proposed', 'random',
    'kernel_02', 'kernel_04', 'kernel_08', 'kernel_16' (bandwidth = trailing numbers * 0.1; for example, 'kernel_02'
    means kernel method with bandwidth 0.2)  

Example:
seed = 30
trans_setting = 'homo'
reward_setting = 'smooth'
gamma = 0.9
N = int(25)
type_est = "proposed"
'''
# seed = int(sys.argv[1])
# trans_setting = sys.argv[2]
# reward_setting = sys.argv[3]
# gamma = float(sys.argv[4])
# N = int(sys.argv[5])
# type_est = str(sys.argv[6])

seed = 29
kappa = 55
num_threads = 5
gamma = 0.9
trans_setting = 'pwconst2'
reward_setting = 'homo'
N = int(25)
RBFSampler_random_state = 1
np.random.seed(seed)
startTime = datetime.now()
np.random.seed(seed)

# criterion of cross validation. Takes value from 'ls' (least squares) or 'kerneldist' (kernel distance)
metric = 'ls'
# grids of hyperparameters of decision trees to search over in cross validation
param_grid = {"max_depth": [3, 5, 6],
              "min_samples_leaf": [50, 60, 70]}
# the type of test statistic to use for detecting change point. Takes values
# in 'int_emp' (integral), '' (unnormalized max), and 'normalized' (normalized max)
method = '_int_emp'
# basis functions. In evaluation, we use decision trees with only linear terms of states
qmodel = 'polynomial'
# degree of the basis function. degree = 1 or 0 for Linear term only
degree = 1
# true change point
time_change_pt_true = int(50)
# number of new individuals to simulate to calculate the discounted reward in infinite horizon
N_factor = 8
# number of new time points to simulate to calculate the discounted reward in infinite horizon
T1_intercept = 200


plot_value = seed < 5

# %% parameters to simulate data
# terminal timestamp
T = 100
# dimension of X0
p = 1
# mean vector of X0
mean0 = 0
# diagonal covariance of X0
cov0 = 0.5
# mean vector of random errors zt
mean = 0
# diagonal covariance of random errors zt
cov = 0.25

# oracle change points and cluster membership
g_index_true = np.append(np.zeros(int(N/3)), np.ones(int(N/3)))
g_index_true = np.append(g_index_true , 2*np.ones(int(N/3)))
#%% environment setup
append_name = '_N' + str(N) + '_1d'
if not os.path.exists('data'):
    os.makedirs('data', exist_ok=True)
data_path = 'data/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) + \
                                             append_name
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
data_path += '/sim_result' + method + '_gamma' + re.sub("\\.", "", str(gamma)) + \
             append_name + '_' + str(seed)
if not os.path.exists(data_path):
    os.makedirs(data_path, exist_ok=True)
os.chdir(data_path)
stdoutOrigin = sys.stdout
sys.stdout = open("log_" + type_est + ".txt", "w")

num_threads = 3
time_terminal = T

#%% generate data for estimating the optimal policy
def simulate(changepoints_true, N=25, optimal_policy_model = None, S0=None, A0=None, T0=0, T1=T_new):
    w = 0.01
    delta = 1 / 10
    States = np.zeros([N, T, p])
    Rewards = np.zeros([N, T-1])
    Actions = np.zeros([N, T-1])
    if trans_setting == 'homo' and reward_setting == 'pwconst2':
        def mytransition_function(t):
            return sim_dat.transition_homo(mean, cov)
        def myreward_function(t):
            return sim_dat.reward_pwconstant2(t)
    elif trans_setting == 'homo' and reward_setting == 'smooth':
        def mytransition_function(t):
            return sim_dat.transition_homo(mean, cov)
        def myreward_function(t):
            return sim_dat.reward_smooth2(t, w)
    elif trans_setting == 'pwconst2' and reward_setting == 'homo':
        def mytransition_function(t):
            return sim_dat.transition_pwconstant2(t, mean, cov)
        def myreward_function(t):
            return sim_dat.reward_homo()
    elif trans_setting == 'smooth' and reward_setting == 'homo':
        def mytransition_function(t):
            return sim_dat.transition_smooth2(t, mean, cov, w)
        def myreward_function(t):
            return sim_dat.reward_homo()
    # for i in range(N):   
    sim_dat = sim.simulate_data(N, T, p, changepoints_true, delta)
    States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function)
        # States[i, :, :] = States0
        # Rewards[i, :] = Rewards0
        # Actions[i, :] = Actions0
    def transform(x):
        return (x - np.mean(x)) / np.std(x)
    for i in range(p):
        States[:,:,i] = transform(States[:,:,i])
    Actions = Actions.astype(int)
    return States, Rewards, Actions


# States, Rewards, Actions = simulate(N=N, optimal_policy_model = None, T0=0, T1=T)
basemodel = DecisionTreeRegressor(random_state=seed)


#%% estimate the value of the estimated policy
rbf_bw = None
def estimate_value(States, Rewards, Actions, param_grid, basemodel, changepoints_true, T1_intercept=10):
    '''
    T1_intercept : the length of time points following optimal policy
        DESCRIPTION. The default is 10.
    '''
    # select decision tree's parameters
    out = select_model_cv(States, Rewards, Actions, param_grid, bandwidth=rbf_bw,
                        qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=2, tol=1e-4,
                        nfold = 3, num_threads = num_threads, metric = metric)
    model = out['best_model']
    print(model)
    q_all = stat.q_learning(States, Rewards, Actions, qmodel, degree, gamma, rbf_bw=rbf_bw)
    q_all_fit = q_all.fit(model, max_iter=200, tol = 1e-6) # q_all will also be updated
    if plot_value:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for a in range(2):
            tree.plot_tree(q_all.q_function_list[a], ax=axs[a])
            axs[a].set_title('Action ' + str(2*a-1), loc='left')
        fig.savefig('plot_policy' + type_est + '.pdf', bbox_inches='tight', pad_inches = 0.5)
        plt.close('all')
        # plt.show()
        
    estimated_value = []
    for changepoint_true in np.unique(changepoints_true):
        N_new = np.sum(changepoints_true == changepoint_true) * N_factor
        _, Rewards_new, _ = simulate(changepoint_true, N_new, optimal_policy_model=q_all, T0=changepoint_true, T1 = changepoint_true + T1_intercept)
        est_v = 0.0
        for t in range(T1_intercept):
            est_v += Rewards_new[:,t] * gamma**t
        estimated_value.append(est_v)
    return estimated_value

#%% 1 overall policy: assume stationarity and homogeniety throughout
if type_est == 'overall':
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_overall = estimate_value(States, Rewards, Actions, param_grid, basemodel=model, changepoints_true=changepoints_true)
    estimated_value_overall = np.mean(estimated_value_overall)
    print("Overall estimated reward:", estimated_value_overall, "\n")
    pickle.dump(estimated_value_overall, open("value_overall_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()
    if plot_value:
        fig = plt.hist(estimated_value_overall, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of overall values')
        plt.savefig("hist_value_overall_gamma" + re.sub("\\.", "", str(gamma)) + ".png")



# %% estimate the oracle policy: piecewise Q function before and after change point
#%% 2 fit the Q model with known change point
if type_est == 'oracle_cp':
    # time_change_pt = time_change_pt_true
    # model = DecisionTreeRegressor(random_state=seed)
    # estimated_value_oracle_cp = estimate_value(States[:, time_change_pt:, :], Rewards[:, time_change_pt:], Actions[:, time_change_pt:], param_grid, model)
    # estimated_value_oracle_cp = np.mean(estimated_value_oracle_cp)
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle_cp = []
    for g in np.unique(g_index_true):
        cp_g = int(changepoints_true[np.where(g_index_true == g)[0][0]])
        estimated_value = estimate_value(States[:,cp_g+1:,:], Rewards[:,cp_g+1:], Actions[:,cp_g+1:], param_grid, model, changepoints_true=cp_g)
        estimated_value_oracle_cp.append(estimated_value)
    estimated_value_oracle_cp = np.mean(estimated_value_oracle_cp)
    print("Oracle estimated reward:", estimated_value_oracle_cp, "\n")
    pickle.dump(estimated_value_oracle_cp, open("value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()
    if plot_value:
        fig = plt.hist(estimated_value_oracle_cp, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")

#%% 3 fit the Q model with known cluster membership
if type_est == 'oracle_cluster':
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle_cluster = []
    for g in np.unique(g_index_true):
        estimated_value = estimate_value(States[g_index_true == g,:,:], Rewards[g_index_true == g,:], Actions[g_index_true == g,:], param_grid, model,changepoints_true=cp_g)
        estimated_value_oracle_cluster.append(estimated_value)
    if plot_value:
        fig = plt.hist(estimated_value_oracle_cluster, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")
    estimated_value_oracle_cluster = np.mean(estimated_value_oracle_cluster)
    print("Oracle estimated reward:", estimated_value_oracle_cluster, "\n")
    pickle.dump(estimated_value_oracle_cluster, open("value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()
    
#%% 4 fit the Q model with known cluster membership and change points
if type_est == 'oracle':
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle = []
    for g in np.unique(g_index_true):
        cp_g = int(changepoints_true[np.where(g_index_true == g)[0][0]])
        estimated_value = estimate_value(States[g_index_true == g, cp_g+1:,:], Rewards[g_index_true == g, cp_g+1:], Actions[g_index_true == g, cp_g+1:], param_grid, model,changepoints_true=cp_g)
        estimated_value_oracle_cluster.append(estimated_value)
    if plot_value:
        fig = plt.hist(estimated_value_oracle_cluster, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")
    estimated_value_oracle = np.mean(estimated_value_oracle)
    print("Oracle estimated reward:", estimated_value_oracle, "\n")
    pickle.dump(estimated_value_oracle_cluster, open("value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()   
    
#%% 5 fit the Q model with last observations and known cluster membership
if type_est == 'last':
    # time_change_pt = time_change_pt_true
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_last = []
    for g in np.unique(g_index_true):
        cp_g = changepoints_true[np.where(g_index_true == g)[0]]
        estimated_value = estimate_value(States[g_index_true == g, -2:,:].reshape((np.sum(g_index_true == g), 2, -1)), Rewards[g_index_true == g, -1].reshape((-1,1)), Actions[g_index_true == g, -1].reshape((-1,1)), param_grid, model, changepoints_true=cp_g)
        # States=States[g_index_true == g, -2:,:].reshape((np.sum(g_index_true == g), 2, -1))
        # Rewards=Rewards[g_index_true == g, -1].reshape((-1,1))
        # Actions=Actions[g_index_true == g, -1].reshape((-1,1))
        estimated_value_last.append(estimated_value)
    if plot_value:
        fig = plt.hist(estimated_value_oracle_cluster, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")
    estimated_value_last = np.mean(estimated_value_last)
    print("Oracle estimated reward:", estimated_value_last, "\n")
    pickle.dump(estimated_value_oracle_cluster, open("value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()   

#%% 6 individual policy learning
if type_est == 'indi':
    # time_change_pt = time_change_pt_true
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_indi  = []
    for i in range(States.shape[0]):
        # cp_g = time_change_pt[np.where(g_index_true == g)[0]]
        estimated_value = estimate_value(States[i,:,:].reshape((1, States.shape[1], -1)), Rewards[i,:].reshape((1, Rewards.shape[1])), Actions[i,:].reshape((1, Rewards.shape[1])), param_grid, model, changepoints_true=changepoints_true[i])
        estimated_value_indi.append(estimated_value)
    if plot_value:
        fig = plt.hist(estimated_value_indi, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")
    estimated_value_indi = np.mean(estimated_value_indi)
    print("Oracle estimated reward:", estimated_value_indi, "\n")
    pickle.dump(estimated_value_indi, open("value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".dat", "wb"))
    sys.stdout.flush()      

