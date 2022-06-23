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
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
# sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
import simu.simulate_data_pd as sim
# from functions.evaluation_separateA import *
from simu.evaluation import *
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
seed = int(sys.argv[1])
is_save = int(sys.argv[2])
# seed=0
# is_save=1
# print(seed)
# print(is_save)
# trans_setting = sys.argv[2]
# reward_setting = sys.argv[3]
# gamma = float(sys.argv[4])
# N = int(sys.argv[5])
# type_est = str(sys.argv[6])

# seed = 0
num_threads = 3
gamma = 0.9
trans_setting = 'pwconst2'
reward_setting = 'homo'
RBFSampler_random_state = 1
np.random.seed(seed)
startTime = datetime.now()
np.random.seed(seed)
# type_est = ['last']
type_est = ['indi']
# criterion of cross validation. Takes value from 'ls' (least squares) or 'kerneldist' (kernel distance)
metric = 'ls'
# grids of hyperparameters of decision trees to search over in cross validation
param_grid = {"max_depth": [3, 5, 6],
              "min_samples_leaf": [50, 60, 70]}
# param_grid = {"max_depth": [5, 10, 20],
#               "min_samples_leaf": [80, 90, 100]}
# the type of test statistic to use for detecting change point. Takes values
# in 'int_emp' (integral), '' (unnormalized max), and 'normalized' (normalized max)
method = '_int_emp'
# basis functions. In evaluation, we use decision trees with only linear terms of states
qmodel = 'polynomial'
# degree of the basis function. degree = 1 or 0 for Linear term only
degree = 1
# true change point
time_change_pt_true = int(50)
# number of new individuals = N * N_factor to simulate to calculate the discounted reward in infinite horizon
N_factor = 8
# number of new time points = changepoint + T1_interval to simulate to calculate the discounted reward in infinite horizon
T1_interval = 200


plot_value = False

# %% parameters to simulate data
N=36
# terminal timestamp
T = 100
# dimension of X0
p = 2
# mean vector of X0
mean0 = 0
# diagonal covariance of X0
cov0 = 0.5
# mean vector of random errors zt
mean = 0
# diagonal covariance of random errors zt
cov = 0.25

mag_factor=3
# oracle change points and cluster membership
# g_index_true = np.append(np.append(np.zeros(int(N/2)), np.ones(int(N/2))))
# changepoints_true = np.append(np.append(89*np.ones(int(N/3)), 79*np.ones(int(N/3))), 69*np.ones(int(N/3)))
coef1=[[mag_factor*-0.08, mag_factor*0, mag_factor*-0.25],[mag_factor*0.08, mag_factor*0, mag_factor*0.25]]
coef =[coef1,coef1[::-1]]# this is acceptable 0609
signal = [[mag_factor*0, mag_factor*0], [mag_factor*0, mag_factor*0]]

#%% environment setup
def setpath(seed):
    os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
    append_name = '_N' + str(N) + '_1d'
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
    data_path = 'data/coef'+str(coef)+'/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) + \
                                                 append_name
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    data_path += '/sim_result' + method + '_gamma' + re.sub("\\.", "", str(gamma)) + \
                 append_name + '_' + str(seed)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)
    return
setpath(seed)
# stdoutOrigin = sys.stdout
# # sys.stdout = open("log_" + type_est + ".txt", "w")
# sys.stdout = open("log_" + ".txt", "w")
# num_threads = 3
# time_terminal = T
def save_data(file_name, estimated_value,estimated_value_path, converge,err):
    setpath(seed)
    if is_save:
        with open(file_name, "wb")as f:
            pickle.dump({'final':estimated_value,
                         'path':estimated_value_path,
                         'converge':converge,
                         'Qerr':err},f)
    return
#%% generate data for estimating the optimal policy
def simulate(i, N_new=25, optimal_policy_model = None, S0=None, A0=None, 
             T0=0, T1=T, path_index=None):
    '''
    simulate data after change points
    '''
    w = 0.01
    delta = 1 / 10
    States = np.zeros([N_new, T1-T0, p])
    Rewards = np.zeros([N_new, T1-T0-1])
    Actions = np.zeros([N_new, T1-T0-1])
    coef_tmp = [None] * 2
    if i==0:
        coef_tmp[0] = coef[0][0]
        coef_tmp[1] = coef[1][0]
        signal_tmp = [signal[0][0], signal[1][0]]
    elif i==1:
        coef_tmp[0] = coef[0][1]
        coef_tmp[1] = coef[1][1]
        signal_tmp = [signal[1][0], signal[1][1]]
 
    sim_dat = sim.simulate_data(N_new, T, p, T0, delta)
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
            return sim_dat.transition_pwconstant2(t, mean, cov, coef_tmp, signal_tmp)
        def myreward_function(t):
            return sim_dat.reward_homo()
    elif trans_setting == 'smooth' and reward_setting == 'homo':
        def mytransition_function(t):
            return sim_dat.transition_smooth2(t, mean, cov, w)
        def myreward_function(t):
            return sim_dat.reward_homo()
    States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function,
                                                T0 = T0, T1 = T1, optimal_policy_model=optimal_policy_model, path_index = path_index)
    Actions = Actions.astype(int)
    return States, Rewards, Actions

def gen_dat(N, T, coef, signal, changepoint_list=None, seed=1):
    np.random.seed(seed)
    if changepoint_list is None:
        changepoint_list = [int(T/2) + int(0.1 * T) - 1,int(T/2) - int(0.1 * T) - 1] 
    w = 0.01
    delta = 1 / 10
    changepoints_true = np.zeros([N, 1])
    States = np.zeros([N, T, p])
    Rewards = np.zeros([N, T-1])
    Actions = np.zeros([N, T-1])
    coef_tmp = [None] * 2
    changepoint = 0
    for i in range(N):
        if i < int(N/2):
            changepoint = changepoint_list[0]
            coef_tmp[0] = coef[0][0]
            coef_tmp[1] = coef[1][0]
            signal_tmp = [signal[0][0], signal[1][0]]

        else:
            changepoint = changepoint_list[1]
            coef_tmp[0] = coef[0][1]
            coef_tmp[1] = coef[1][1]
            signal_tmp = [signal[0][1], signal[1][1]]
            
        sim_dat = sim.simulate_data(1, T, p, changepoint, delta)
        # print(trans_setting, reward_setting)
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
                return sim_dat.transition_pwconstant2(t, mean, cov, coef_tmp, signal_tmp)
            def myreward_function(t):
                return sim_dat.reward_homo()
        elif trans_setting == 'smooth' and reward_setting == 'homo':
            def mytransition_function(t):
                return sim_dat.transition_smooth2(t, mean, cov, w)
            def myreward_function(t):
                return sim_dat.reward_homo()
        States0, Rewards0, Actions0 = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function)
        States[i, :, :] = States0
        Rewards[i, :] = Rewards0
        Actions[i, :] = Actions0
        changepoints_true[i, ] = changepoint
    # normalize state variables
    def transform(x):
        return (x - np.mean(x)) / np.std(x)
    for i in range(p):
        States[:,:,i] = transform(States[:,:,i])
    g_index_true = np.hstack([np.zeros(int(N/2)), np.ones(int(N/2))])
    Actions = Actions.astype(int)
    g_index_true = g_index_true.astype(int)
    return States, Rewards, Actions, changepoints_true, g_index_true

States, Rewards, Actions, changepoints_true, g_index_true = gen_dat(N, T, 
                                                      coef, signal,None,seed + 100)
indexes = np.unique(changepoints_true, return_index=True)[1]
changepoints_unique = [changepoints_true[index].item() for index in sorted(indexes)]
basemodel = DecisionTreeRegressor(random_state=seed)

#%% estimate the value of the estimated policy
rbf_bw = None
def estimate_value(States, Rewards, Actions, param_grid, basemodel, cp_run, T1_interval=10):
    '''
    T1_interval : the length of time points following optimal policy
        DESCRIPTION. The default is 10.
    '''
    # select decision tree's parameters
    out = select_model_cv(States, Rewards, Actions, param_grid, bandwidth=rbf_bw,
                        qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                        nfold = 5, num_threads = num_threads, metric = metric, num_basis=2)
    model = out['best_model']
    # print(model)
    q_all = stat.q_learning(States, Rewards, Actions, qmodel, 2, gamma, rbf_bw=rbf_bw)
    q_all_fit = q_all.fit(model, max_iter=200, tol = 1e-6) # q_all will also be updated
    if plot_value:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for a in range(2):
            tree.plot_tree(q_all.q_function_list[a], ax=axs[a])
            axs[a].set_title('Action ' + str(2*a-1), loc='left')
        fig.savefig('plot_policy' + '.pdf', bbox_inches='tight', pad_inches = 0.5)
        plt.close('all')
        # plt.show()
    # estimated_value = []
    # cp_list = np.unique(cp_run)
    # for changepoint_true in cp_list:
    #     N_new = np.sum(changepoints_true == changepoint_true) * N_factor
    #     _, Rewards_new, _, = simulate(np.where(changepoints_unique==changepoint_true)[0][0], N_new,optimal_policy_model=q_all, T0=int(changepoint_true), T1 = int(changepoint_true + T1_interval))
    #     est_v = 0.0
    #     for t in range(T1_interval):
    #         est_v += Rewards_new[:,t] * gamma**t
    #     estimated_value.append(est_v)
        
    cp_list = np.unique(cp_run)
    # print('qlen', len(q_all.q_function_list_path), q_all)
    last_ind = int(min((200/10)+1, q_all_fit[2][1]/10))
    estimated_path = [None] * int((200/10+1))
    # print('last_ind', last_ind)
    for path_index in range(last_ind):
        # print('pi', path_index, 'qlist', q_all.q_function_list_path[path_index])
        estimated_value = []
        # print('cp_list',cp_list)
        for changepoint_true in cp_list:
            # print('changepoint_true',changepoint_true)
            N_new = np.sum(changepoints_true == changepoint_true) * N_factor
            _, Rewards_new, _, = simulate(np.where(changepoints_unique==changepoint_true)[0][0], 
                                          N_new,optimal_policy_model=q_all, T0=int(changepoint_true), T1 = int(changepoint_true + T1_interval), path_index=path_index)
            est_v = 0.0
            for t in range(T1_interval):
                est_v += Rewards_new[:,t] * gamma**t
            estimated_value = estimated_value + list(est_v)
        # if path_index==0:
            # print('estimated_value',estimated_value)
        estimated_path[path_index] = estimated_value

    return estimated_path[last_ind-1].copy(), estimated_path.copy(), int(q_all_fit[2][2]),q_all_fit[2][0][-1]

def calculate_path_value(estimated_value_path):
    estimated_value = [None] * len(estimated_value_path)
    for path_index in range(len(estimated_value_path)):
        # print('pi',path_index)
        # if np.array(estimated_value_path[path_index]).any(None):
        if estimated_value_path[path_index] is None:
            break
        elif None in estimated_value_path[path_index]:
            break
        else:
            estimated_value[path_index] = np.mean(estimated_value_path[path_index])
    return estimated_value
#%% 1 overall policy: assume stationarity and homogeniety throughout
startTime = datetime.now()
if 'overall' in type_est:
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_overall, estimated_value_overall_path,converge, err = estimate_value(States, Rewards, Actions, param_grid, basemodel=model, cp_run=changepoints_true)
    if plot_value:
        fig = plt.hist(estimated_value_overall, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of overall values')
        plt.savefig("hist_value_overall_gamma" + re.sub("\\.", "", str(gamma)) + ".png")
    estimated_value_overall = np.mean(estimated_value_overall)
    print("Overall estimated reward:", estimated_value_overall, 'converge',converge,'err',err, "\n")
    file_name = "value_overall_gamma_overall" + re.sub("\\.", "", str(gamma)) + ".dat"
    save_data(file_name, estimated_value_overall, calculate_path_value(estimated_value_overall_path),converge,err)
    sys.stdout.flush()
runtimeone = datetime.now() - startTime
print(runtimeone)

# %% estimate the oracle policy: piecewise Q function before and after change point
#%% 2 fit the Q model with known change point
if 'oracle_cp' in type_est:
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle_cp = []
    estimated_value_oracle_cp_path = [[] for i in range(int(200/10+1))]
    States_stack = np.zeros([N*T, p])
    Rewards_stack = np.zeros([1, N*T])
    Actions_stack = np.zerors([1, N*T])
    T_tmp = 0
    T_mark =0
    for g in np.unique(g_index_true):
        cp_g = int(changepoints_true[np.where(g_index_true == g)[0][0]])
        # T_tmp = T - cp_g - 1
        States_tmp =  States[np.where(g_index_true == g)[0], cp_g:, :].transpose(2, 0, 1).reshape(States.shape[2], -1).T
        Rewards_tmp = Rewards[np.where(g_index_true == g)[0], cp_g:].flatten()
        Actions_tmp = Actions[np.where(g_index_true == g)[0], cp_g:].flatten()
        if g == np.unique(g_index_true)[0]:
            States_stack = States_tmp.copy()
            Rewards_stack = Rewards_tmp.copy()
            Actions_stack = Actions_tmp.copy()
        else:
            States_stack = np.vstack([States_stack, States_tmp])
            Rewards_stack = np.append(Rewards_stack, Rewards_tmp)
            Actions_stack = np.append(Actions_stack, Actions_tmp)
    for g in np.unique(g_index_true):
        cp_g = int(changepoints_true[np.where(g_index_true == g)[0][0]])
        estimated_value,estimated_value_path = estimate_value(States[:,cp_g+1:,:], Rewards[:,cp_g+1:], Actions[:,cp_g+1:], param_grid, model, cp_run=cp_g)
        estimated_value_oracle_cp.append(estimated_value)
        for path_index in range(int(200/10+1)):
            estimated_value_oracle_cp_path[path_index].append(estimated_value_path[path_index])
        
    if plot_value:
        fig = plt.hist(estimated_value_oracle_cp, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")
    estimated_value_oracle_cp = np.mean(estimated_value_oracle_cp)
    print("Oracle cp estimated reward:", estimated_value_oracle_cp, "\n")
    file_name = "value_oracle_gamma_oracle_cp" + re.sub("\\.", "", str(gamma)) + ".dat"
    save_data(file_name, estimated_value_oracle_cp, calculate_path_value(estimated_value_oracle_cp_path))
    # if is_save:
    #     with open("value_oracle_gamma_oracle_cp" + re.sub("\\.", "", str(gamma)) + ".dat", "wb") as f:
    #         pickle.dump(estimated_value_oracle_cp, f)
    sys.stdout.flush()
#%% 3 fit the Q model with known cluster membership
if  'oracle_cluster' in type_est:
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle_cluster = []
    estimated_value_oracle_cluster_path = [[] for i in range(int(200/10+1))] 
    converge_list =[]
    err_list=[]
    for g in np.unique(g_index_true):
        # print('g',g)
        cp_g = int(changepoints_true[np.where(g_index_true == g)[0][0]])
        estimated_value,estimated_value_path,converge,err = estimate_value(States[g_index_true == g,:,:], Rewards[g_index_true == g,:], Actions[g_index_true == g,:], param_grid, model,cp_run=cp_g)
        estimated_value_oracle_cluster.append(estimated_value)
        converge_list.append(converge)
        err_list.append(err)
        for path_index in range(int(200/10+1)):
            estimated_value_oracle_cluster_path[path_index].append(estimated_value_path[path_index])

    if plot_value:
        fig = plt.hist(np.hstack(estimated_value_oracle_cluster), bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")
    estimated_value_oracle_cluster = np.mean(estimated_value_oracle_cluster)
    converge=np.mean(converge_list)
    err= np.mean(err_list)
    print("Oracle cluster estimated reward:", estimated_value_oracle_cluster,'converge',converge,'err',err, "\n")
    file_name="value_oracle_gamma_oracluster" + re.sub("\\.", "", str(gamma)) + ".dat"
    save_data(file_name, estimated_value_oracle_cluster, calculate_path_value(estimated_value_oracle_cluster_path), converge, err)
    # if is_save:
    #     with  open("value_oracle_gamma_oracluster" + re.sub("\\.", "", str(gamma)) + ".dat", "wb") as f:
    #         pickle.dump(estimated_value_oracle_cluster, f)
    sys.stdout.flush()
    
#%% 4 fit the Q model with known cluster membership and change points
startTime = datetime.now()
if 'oracle' in type_est:
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_oracle = []
    estimated_value_oracle_path =  [[] for i in range(int(200/10+1))]
    converge_list =[]
    err_list=[]
    for g in np.unique(g_index_true):
        cp_g = int(changepoints_true[np.where(g_index_true == g)[0][0]])
# States=States[g_index_true == g, cp_g+1:,:]
# Rewards = Rewards[g_index_true == g, cp_g+1:]
# Actions = Actions[g_index_true == g, cp_g+1:]
        estimated_value,estimated_value_path,converge,err = estimate_value(States[g_index_true == g, cp_g+1:,:], Rewards[g_index_true == g, cp_g+1:], Actions[g_index_true == g, cp_g+1:], param_grid, model,cp_run=cp_g)
        estimated_value_oracle.append(estimated_value)
        converge_list.append(converge)
        err_list.append(err)
        for path_index in range(int(200/10+1)):
            estimated_value_oracle_path[path_index].append(estimated_value_path[path_index])

    if plot_value:
        fig = plt.hist(estimated_value_oracle_cluster, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")
    estimated_value_oracle = np.mean(estimated_value_oracle)
    converge=np.mean(converge_list)
    err= np.mean(err_list)
    print("Oracle estimated reward:", estimated_value_oracle, 'converge',converge,'err',err, "\n")
    file_name = "value_oracle_gamma_oracle" + re.sub("\\.", "", str(gamma)) + ".dat"
    save_data(file_name, estimated_value_oracle,  calculate_path_value(estimated_value_oracle_path),converge,err)
    sys.stdout.flush()   
runtimeone = datetime.now() - startTime
print(runtimeone)
#%% 5 fit the Q model with last observations and known cluster membership
if 'last' in type_est:
    # time_change_pt = time_change_pt_true
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_last = []
    estimated_value_last_path = [[] for i in range(int(200/10+1))] 
    converge_list =[]
    err_list=[]
    for g in np.unique(g_index_true):
        cp_g = changepoints_true[np.where(g_index_true == g)[0]]
        estimated_value,estimated_value_path,converge,err = estimate_value(States[g_index_true == g, -2:,:].reshape((np.sum(g_index_true == g), 2, -1)), Rewards[g_index_true == g, -1].reshape((-1,1)), Actions[g_index_true == g, -1].reshape((-1,1)), param_grid, model, cp_run=cp_g)
        estimated_value_last.append(estimated_value)
        converge_list.append(converge)
        err_list.append(err)
        for path_index in range(int(200/10+1)):
            # print('pi', path_index,'estimated_value_last_path[path_index]',estimated_value_last_path[path_index])
            # print('estimated_value_path[path_index]',estimated_value_path[path_index])
            estimated_value_last_path[path_index].append(estimated_value_path[path_index])
            # print('estimated_value_last_path[path_index]',estimated_value_last_path[path_index])
            # estimated_value_last_path[path_index].append(path_index)

    if plot_value:
        fig = plt.hist(estimated_value_oracle_cluster, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")
    estimated_value_last = np.mean(estimated_value_last)
    converge=np.mean(converge_list)
    err= np.mean(err_list)
    print("last estimated reward:", estimated_value_last,'converge',converge,'err',err, "\n")
    file_name = "value_oracle_gamma_last" + re.sub("\\.", "", str(gamma)) + ".dat"
    save_data(file_name, estimated_value_last, calculate_path_value(estimated_value_last_path),converge, err)
    # if is_save:
    #     with  open("value_oracle_gamma_last" + re.sub("\\.", "", str(gamma)) + ".dat", "wb") as f:
    #         pickle.dump(estimated_value_last, f)
    sys.stdout.flush()   

#%% 6 individual policy learning
startTime = datetime.now()
if 'indi' in type_est:
    model = DecisionTreeRegressor(random_state=seed)
    estimated_value_indi  = []
    estimated_value_indi_path =  [[] for i in range(int(200/10+1))] 
    converge_list =[]
    err_list=[]
    for i in range(States.shape[0]):
        # cp_g = time_change_pt[np.where(g_index_tr
    # time_change_pt = time_change_pt_truue == g)[0]]
        estimated_value, estimated_value_path,converge,err = estimate_value(States[i,:,:].reshape((1, States.shape[1], -1)), Rewards[i,:].reshape((1, Rewards.shape[1])), Actions[i,:].reshape((1, Rewards.shape[1])), param_grid, model, cp_run=changepoints_true[i])
        estimated_value_indi.append(estimated_value)
        converge_list.append(converge)
        err_list.append(err)
        for path_index in range(int(200/10+1)):
            estimated_value_indi_path[path_index].append(estimated_value_path[path_index])
    if plot_value:
        fig = plt.hist(estimated_value_indi, bins = 50)
        plt.xlabel('Values')
        plt.ylabel('Count')
        plt.title('Distribution of oracle values')
        plt.savefig("hist_value_oracle_gamma" + re.sub("\\.", "", str(gamma)) + ".png")
    estimated_value_indi = np.mean(estimated_value_indi)
    converge=np.mean(converge_list)
    err= np.mean(err_list)
    print("indi estimated reward:", estimated_value_indi,'converge',converge,'err',err, "\n")
    file_name="value_oracle_gamma_indi" + re.sub("\\.", "", str(gamma)) + ".dat"
    save_data(file_name, estimated_value_indi, calculate_path_value(estimated_value_indi_path),converge,err)
    # if is_save:
    #     with open("value_oracle_gamma_indi" + re.sub("\\.", "", str(gamma)) + ".dat", "wb") as f:
    #         pickle.dump(estimated_value_indi, f)
    sys.stdout.flush()      
runtimeone = datetime.now() - startTime
print(runtimeone)
