# -*- coding:utf-8 -*-
'''
Simulate non-stationary time series data and apply Q-learning.
estimate the optimal reward after identifying change point
'''
#!/usr/bin/python
import  sys, os,  pickle
from copy import copy
from sklearn.preprocessing import PolynomialFeatures
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the "ClusterRL" directory
cluster_rl_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Append the ClusterRL directory to sys.path
sys.path.append(cluster_rl_dir)

import numpy as np
from datetime import datetime
from functions.simulate_data_1d_flexible import simulate
from sklearn.tree import DecisionTreeRegressor
import functions.compute_test_statistics_separateA as stat
from functions.evaluation import select_model_cv
import functions.simu_mean_detect as mean_detect
from joblib import Parallel, delayed
path_name = os.getcwd()
#%%
# argv should be: seed, kappa, degree, num_threads
seed = int(sys.argv[1])
type_est = str(sys.argv[2])
N_per_cluster = int(sys.argv[3])
T_new = int(sys.argv[4])
trans_setting = sys.argv[5]
nthread = int(sys.argv[6])
cov = float(sys.argv[7])
cp_detect_interval = int(sys.argv[8])
is_tune_parallel = int(sys.argv[9])
is_cp_parallel= int(sys.argv[10])
max_iter=int(sys.argv[11])
effect_size_factor =float(sys.argv[12])
env_type=sys.argv[13]
K_true=int(sys.argv[14])
stationary_team_dynamic=int(sys.argv[15])
T_initial=int(sys.argv[16])
fixed_kappamin=float(sys.argv[17])
true_cp_interval=int(sys.argv[18])
reward_type=sys.argv[19]
is_transform=int(sys.argv[20])
refit=int(sys.argv[21])
stationary_changeactionsign=int(sys.argv[22])
is_r_orignial_scale=int(sys.argv[23])
print('type_est', type_est)
print('trans_setting', trans_setting)
np.random.seed(seed)

# %% simulate data
reward_setting = 'homo'
gamma = 0.9

Kl_fun='Nlog(NT)/T' #"sqrtN"
startTime = datetime.now()

np.random.seed(seed)
threshold_type = "maxcusum"
early_stopping=1
p = 3

K_list = list(range(1,5))

qmodel = 'polynomial'
degree = 1
C =1 #1 # if tune K in iteration: np.log(N*cp_detect_interval)
#%% environment for saving online values
def setpath():
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
    if type_est in ["proposed", "only_clusters", "cluster_atend", "cluster_atend2"
                    ]:
        if Kl_fun=='Nlog(NT)/T':
            Kl_fun_name = "Nlog(NT)_T"
        else:
            Kl_fun_name = Kl_fun
        method_name = type_est + str(K_list)+'/Kl_fun'+Kl_fun_name+'_C'+str(C)
        if type_est == "proposed":
            method_name += "earlystop"+str(early_stopping)+'/max_iter'+str(max_iter)
    else:
        method_name  = type_est
    if type_est in ["proposed", "only_cp",# "cp_indi"
                    ]:
        if fixed_kappamin < 1:
            method_name += '/kappamin'+str(fixed_kappamin)
    if is_transform != 1 and type_est in ["proposed", "only_clusters", "only_cp",
                                           "cluster_atend", "cluster_atend2" #"cp_indi"
                                          ]:
        method_name +='/transform'+str(is_transform)
        
    if effect_size_factor != 1.0:
        trans_setting_name =trans_setting + "/effect_size"+str(effect_size_factor)
    else:
        trans_setting_name = trans_setting
    if env_type != "randomstationary":
        trans_setting_name += '/'+env_type
    if reward_type != "next_state":
        trans_setting_name += '/reward'+reward_type
    if K_true!=3:
        trans_setting_name+='K_true'+str(K_true)
    if stationary_team_dynamic!=4 and K_true>2:
        trans_setting_name+= '/stationary'+str(stationary_team_dynamic)
        if stationary_team_dynamic<1 and stationary_changeactionsign!=1:
            trans_setting_name+= "changesign"
    
    data_path = 'results/trans' + trans_setting_name +'/N' + str(N_per_cluster) +'/Tnew_' +\
                str(T_new)+'_type_'+method_name+'_cpitv'+ str(cp_detect_interval)+'/true_cp'+str(true_cp_interval)+\
                '/cov'+str(cov) + '/seed'+str(seed) 
                
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)
    
setpath()
print(os.getcwd())
sys.stdout.flush()

if is_r_orignial_scale:
    file_name = 'seed_'+str(seed)+'_originalR.dat'
else:
    file_name = 'seed_'+str(seed)+'.dat'
    
if os.path.exists(file_name) and not refit:
    exit()
    
stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "w")
print("\nName of Python script:", sys.argv)
sys.stdout.flush()


#%%
# normalize state variables
def transform(x):
    return (x - np.mean(x)) / np.std(x)
def transform_back(x):
    # Perform the transformation
    val = (x * 4.141009340555169 + 19.343480280896884) ** 3
    
    # Check for np.inf in the result
    if np.isinf(val).any():
        # Find the indices where the result is infinite
        inf_indices = np.where(np.isinf(val))
        # Return the corresponding x values
        inf_values = x[inf_indices]
        print("The following x values cause the transformation to return np.inf:")
        print(inf_values)
    
    return val

def load_env():
    dat_path = os.path.abspath(os.path.join(path_name, '..'))  
    if env_type == "original":
        env_path = dat_path +'/env/re_learnt_env.dat'
    elif env_type[:16] == 'randomstationary':
        env_path = dat_path +'/env/re_learnt_env_randomstationary.dat'
    else:
        env_path = dat_path +'/env/re_learnt_env_'+env_type+'.dat'
    print('env_path ',env_path )
    if os.path.exists(env_path):
        with open(env_path,'rb') as f:
            env = pickle.load(f)
    else:
        raise ValueError('Env not exist!')
    
    return env

def generate_data(env):

    if env_type == "split_secondcluster":
        # cp_cluster = [[int(T_initial/2)], [round_to_nearest(int(T_initial/3)).item(), round_to_nearest(int(2*T_initial/3)).item()], [0]]

        transition_type = [[0, 1], [2, 3, 4], [stationary_team_dynamic, stationary_team_dynamic]]
    elif env_type == "original":
        # cp_cluster = [[int(T_initial/2)], [round_to_nearest(int(T_initial/3)).item(), round_to_nearest(int(2*T_initial/3)).item()], [0]]

        transition_type = [[0, 1], [0, 2, 3], [stationary_team_dynamic, stationary_team_dynamic]]
    elif env_type=='randomstationary':
        transition_type = [[0, 1], [2, 3], [stationary_team_dynamic, stationary_team_dynamic]]
        cp_cluster = [[int(T_initial/2)], [int(T_initial/2)], [0]]
    elif env_type in ["randomstationary_changesign", "randomstationary_changeactionsign"]:
        transition_type = [[-1, 1], [-3, 3], [stationary_team_dynamic, stationary_team_dynamic]]
        cp_cluster = [[int(T_initial/2)], [int(T_initial/2)], [0]]
    elif env_type in ["randomstationary3_changesign", "randomstationary3_changeactionsign"]:
        transition_type = [[3, -3], [-3, 3], [stationary_team_dynamic, stationary_team_dynamic]]
        cp_cluster = [[int(T_initial/2)], [int(T_initial/2)], [0]]
    else:
        # cp_cluster = [[int(T_initial/2)], [round_to_nearest(int(T_initial/3)).item(), round_to_nearest(int(2*T_initial/3)).item()], [0]]
        transition_type = [[3, 4], [0, 1, 2], [stationary_team_dynamic, stationary_team_dynamic]]
    
    last_trans = [sublist[-1] for sublist in transition_type[:K_true]]
    # Get the unique values in the order they appear in last_trans
    unique_last_trans = list(dict.fromkeys(last_trans))
    g_index_true = np.repeat([unique_last_trans.index(i) for i in last_trans], N_per_cluster)
    cp_dict ={trans: np.max([cp_cluster[i][-1] for i in range(K_true) if last_trans[i] == trans])
              for trans in list(set(last_trans))}
    
    
    def transition_generator(env, g):
        def transition_function(St, At, t):
            St = np.atleast_2d(St)
            At = np.atleast_2d(At)
            poly = PolynomialFeatures(2, interaction_only=True)
            X = poly.fit_transform(np.hstack((At,St)))
            X = np.delete(X, np.s_[2 + p*2:X.shape[1]], 1)
            if g<0:
                if env_type[-10:] =="changesign":
                    re_coef  = -1*env["transition"][int(-1*g)]
                elif stationary_changeactionsign or env_type[-16:] =="changeactionsign":
                    re_coef  = env["transition"][int(-1*g)]
                    re_coef[:,0] = -1*re_coef[:,0] 
                    re_coef[:,-p:] =-1*re_coef[:,0] 
                else:
                    re_coef  = -1*env["transition"][int(-1*g)]
            else:
                re_coef = env["transition"][g]
            # print(re_coef)
            coef = effect_size_factor*re_coef + env["beta_0_transition"]
            return X.dot(coef.T)+ np.random.multivariate_normal(np.zeros(p), cov*np.eye(p))
        return transition_function
    
    def reward_generator(env, g):
        if reward_type == "next_state":
            def reward_function(St, At, t):
                St = np.atleast_2d(St)
                At = np.atleast_2d(At)
                poly = PolynomialFeatures(2, interaction_only=True)
                X = poly.fit_transform(np.hstack((At,St)))
                X = np.delete(X, np.s_[2 + p*2:X.shape[1]], 1)
                if g<0:
                    if env_type[-10:] =="changesign":
                        re_coef  = -1*env["transition"][int(-1*g)]
                    elif stationary_changeactionsign or env_type[-16:] =="changeactionsign":
                        re_coef  = env["transition"][int(-1*g)]
                        re_coef[:,0] = -1*re_coef[:,0] 
                        re_coef[:,-p:] =-1*re_coef[:,0] 
                    else:
                        re_coef  = -1*env["transition"][int(-1*g)]
                        # re_coef  = env["transition"][int(-1*g)]
                        # re_coef[:,0] = -1*re_coef[:,0] 
                        # re_coef[:,-p:] =-1*re_coef[:,0] 
                else:
                    re_coef = env["transition"][g]
                
                coef = effect_size_factor*re_coef + env["beta_0_transition"]
                # print('g', g, 're', re_coef, '/n coef', coef[0,:])
                return X.dot(coef[0,:].T)
        elif reward_type == "current_state":
            def reward_function(St, At, t):
                St = np.atleast_2d(St)
                return St[:,0]
        else:
            raise NotImplementedError()
        return reward_function
    
    
    system_settings_pool = [
        {'N': 1, 'T': T_initial-1, 'p':p,
                           'changepoints': cp_cluster[k],
                           'state_functions':[transition_generator(env, transition_type[k][i]) for i in range(len(cp_cluster[k])+1)], # [transition_generator(env, transition_type[1][0]), transition_generator(env, transition_type[1][1])],
                           'reward_functions': [reward_generator(env, transition_type[k][i]) for i in range(len(cp_cluster[k])+1)], #[reward_generator(env, transition_type[1][0]), reward_generator(env, transition_type[1][1])],
                           'state_change_type': trans_setting,#,
                           'reward_change_type': trans_setting, #'homogeneous',
                           'delta':0.1
                            }
        for k in range(K_true)
        ]
        
    
    States = np.zeros([N_per_cluster*K_true, T_initial, p])
    Rewards = np.zeros([N_per_cluster*K_true, T_initial-1])
    Actions = np.zeros([N_per_cluster*K_true, T_initial-1])
    changepoints_true = np.repeat(0, N_per_cluster*K_true)
    
    def simulate_and_store(k, i, c):
        system_settings = system_settings_pool[k]
        States0, Rewards0, Actions0 = simulate(system_settings, mean0=np.zeros(p), cov0=np.eye(p),
                                               burnin=100)
        return c, States0, Rewards0, Actions0, int(cp_dict[last_trans[k]]) #, int(system_settings['changepoints'][-1])
    
    results = Parallel(n_jobs=-1)(delayed(simulate_and_store)(k, i, k * N_per_cluster + i) for k in range(K_true) for i in range(N_per_cluster))
    
    for c, States0, Rewards0, Actions0, changepoint in results:
        States[c, :, :] = States0
        Rewards[c, :] = Rewards0
        Actions[c, :] = Actions0
        changepoints_true[c] = changepoint
    
    print('finish data generating')
    system_indicator = {0:0,
                        1:1,
                        2:2}
    system_settings_list = [
        {'N': 1, 'T': cp_detect_interval,'p':p,
                           'changepoints': [0],
                           'state_functions': [transition_generator(env, transition_type[1][-1]), transition_generator(env, transition_type[0][-1])],
                           'reward_functions': [reward_generator(env, transition_type[1][-1]), reward_generator(env, transition_type[0][-1])],
                           'state_change_type': trans_setting,#,
                           'reward_change_type': trans_setting,
                           'delta':0.1
                            },
        {'N': 1, 'T': cp_detect_interval,'p':p,
                           'changepoints': [0],
                           'state_functions': [transition_generator(env, transition_type[0][-1]), transition_generator(env, transition_type[1][-1])],
                           'reward_functions': [reward_generator(env, transition_type[0][-1]), reward_generator(env, transition_type[1][-1])],
                           'state_change_type': trans_setting,#,
                           'reward_change_type': trans_setting,
                           'delta':0.1
                            },
        {'N': 1, 'T': cp_detect_interval,'p':p,
                           'changepoints': [0],
                           'state_functions': [transition_generator(env, transition_type[2][-1]), transition_generator(env, transition_type[2][-1])],
                           'reward_functions': [reward_generator(env, transition_type[2][-1]), reward_generator(env, transition_type[2][-1])],
                           'state_change_type': trans_setting,#,
                           'reward_change_type': trans_setting,
                           'delta':0.1
                            }
        ]
    
    if is_r_orignial_scale:
        Rewards = transform_back(Rewards)
    
    return States, Rewards, Actions, changepoints_true, g_index_true, system_indicator, system_settings_list, cp_cluster, last_trans

def find_first_batch_proposed_cp():
    if type_est == "proposed":
        init = "tuneK_iter"
    else:
        init = type_est
    if init in ["tuneK_iter", "only_clusters", "cluster_atend"]:
        init_name = init+ Kl_fun.replace("/", "_")+"_C"+str(C)+'/max_iter'+str(max_iter)+'_K_list'+str(K_list)
    else:
        init_name = init
    # init_name = "tuneK_iter"+ Kl_fun.replace("/", "_")+"_C"+str(C)+'/max_iter'+str(max_iter)+'_K_list'+str(K_list)
    if is_transform !=0:
        init_name += '/statetrans'
    # init_name += '/statetrans'
    trans_setting_name = trans_setting + str(stationary_team_dynamic)
    if env_type != "original":
        trans_setting_name +='/'+env_type
    kappa_max=int(0.9*T_initial)
    kappa_min = int(0.1*T_initial)
    kappa_interval=int(T_initial/50)
    offline_dir = os.path.abspath(os.path.join(path_name, '..', 'offline'))
    file_name = offline_dir +'/results/trans' + trans_setting_name +'/effect_size_factor_' + str(effect_size_factor)+\
        '/Ncluster' + str(N_per_cluster) +'_T' + str(T_initial)+'/K'+ str(K_true)+'/cov'+str(cov)+'/init_'+init_name+\
            '/kappa_max'+str(kappa_max)+'_min'+str(kappa_min)+'_step'+str(kappa_interval)+\
                 '/seed'+str(seed) +"/seed"+str(seed)+".dat"
    print(file_name)
    sys.stdout.flush()
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            dat=pickle.load(f)
        print('find prefitted results!')
        return dat["changepoints"], dat['clustering']
    else:
        False

def generate_data_2018():
    raise NotImplementedError()
    base_transition = np.array([[10, 0.4, -0.04, 0.1],
                                [11, -0.4, 0.05, 0.4],
                                [1.2, -0.02, 0.03, 0.8]])
    state_coef_dict = {
        0:np.zeros(shape=(3, 4)),
        1: 
            # -1*np.array([[-1.0, 1.0, 1, 1],
        #       [-0.5, 1.0, 1.0, 1.0],
        #       [-0.2, 1, 1.0, 1.0]]),
            np.zeros(shape=(3, 4)),
        2:
            # np.array([[-1.0, 1.0, 1, 1],
            #   [-0.5, 1.0, 1.0, 1.0],
            #   [-0.2, 1, 1.0, 1.0]]),
            np.array([[-1.0, 0.0, 0, 0],
                  [-0.5, 0.0, 0.0, 0.0],
                  [-0.2, 0, 0.0, 0.0]]),
        3: 
            # -1*np.array([[-1.0, 1.0, 1, 1],
        #       [-0.5, 1.0, 1.0, 1.0],
        #       [-0.2, 1, 1.0, 1.0]]),
            np.array([[1.0, 0.15, -0.01, 0.02],
                    [1.0, -0.15, 0.01, 0.1],
                    [0.3, -0.01, 0.01, -0.15]]),
        4: 
            # np.array([[-1.0, 1.0, 1, 1],
            #   [-0.5, 1.0, 1.0, 1.0],
            #   [-0.2, 1, 1.0, 1.0]]),
            np.array([[1.0, 0.15, -0.01, 0.02],
                    [1.0, -0.15, 0.01, 0.1],
                    [0.3, -0.01, 0.01, -0.15]]) ,
        5: 
            # np.array([[1.5, 1.05, 1, 1],
            #        [0.5, -0.2, 1.0, 1.0],
            #        [0.2, 1, 1.0, 1.0]])
            np.array([[1.5, 0.05, 0, 0],
                    [0.5, -0.2, 0.0, 0.0],
                    [0.2, 0, 0.0, 0.0]])
        }
    # state_action_coef_dict =state_coef_dict
    state_action_coef_dict = {
        0:np.array([[0.6, 0.3, 0, 0],
          [-0.4, 0, 0, 0],
          [-0.5, 0, 0, 0]]),
        1:np.array([[-0.6, -0.3, 0, 0],
          [0.4, 0, 0, 0],
          [0.5, 0, 0, 0]]) ,
        2:np.array([[0.5, 0.3, 0, 0],
                        [-0.3, 0, 0, 0],
                        [-0.4, 0, 0, 0]]),
        3: np.array([[-0.7, -0.2, 0, 0],
                        [0.5, 0, 0, 0],
                        [0.6, 0, 0, 0]]),
        4:np.array([[0.7, 0.2, 0, 0],
                        [-0.5, 0, 0, 0],
                        [-0.6, 0, 0, 0]]),
        5:np.array([[0.55, 0.25, 0, 0],
                        [-0.4, 0, 0, 0],
                        [-0.5, 0, 0, 0]])
        }

    cp_cluster = [[11], [9, 17], [0]]

    transition_type = [[0, 1], [2,3,4], [5,5]]


    
    last_trans = [sublist[-1] for sublist in transition_type[:K_true]]
    g_index_true = np.repeat([list(set(last_trans)).index(i) for i in last_trans], N_per_cluster)
    cp_dict ={trans: np.max([cp_cluster[i][-1] for i in range(K_true) if last_trans[i] == trans])
              for trans in list(set(last_trans))}
    
    
    def transition_generator(g):
        def transition_function(St, At, t):
            # St = np.atleast_2d(St)
            # At = np.atleast_2d(At)
            St_full = np.insert(St, 0, 1, axis=0)
            ns= (base_transition + state_coef_dict[g] * effect_size_factor +\
                    At * state_action_coef_dict[g] * effect_size_factor) @ St_full 
            # print(ns)
            return ns+np.random.multivariate_normal(np.zeros(p), cov*np.eye(p))
        return transition_function
    
    def reward_generator(g):
        def reward_function(St, At, t):
            # St_full = np.insert(St, 0, 1, axis=0)
            # return (base_transition[0] + state_coef_dict[g][0] * effect_size_factor +\
            #         At * state_action_coef_dict[g][0] * effect_size_factor) @ St_full 
            ## TODO: next state?
            return St[0]
        return reward_function
    
    
    system_settings_list = [
        {'N': 1, 'T': T_initial-1, 'p':p,
                           'changepoints': cp_cluster[k],
                           'state_functions':[transition_generator(transition_type[k][i]) for i in range(len(cp_cluster[k])+1)], # [transition_generator(env, transition_type[1][0]), transition_generator(env, transition_type[1][1])],
                           'reward_functions': [reward_generator(transition_type[k][i]) for i in range(len(cp_cluster[k])+1)], #[reward_generator(env, transition_type[1][0]), reward_generator(env, transition_type[1][1])],
                           'state_change_type': trans_setting,#,
                           'reward_change_type': trans_setting, #'homogeneous',
                           'delta':0.1
                            }
        for k in range(K_true)
        ]
        
    
    States = np.zeros([N_per_cluster*K_true, T_initial, p])
    Rewards = np.zeros([N_per_cluster*K_true, T_initial-1])
    Actions = np.zeros([N_per_cluster*K_true, T_initial-1])
    changepoints_true = np.repeat(0, N_per_cluster*K_true)
    
    def simulate_and_store(k, i, c):
        system_settings = system_settings_list[k]
        States0, Rewards0, Actions0 = simulate(system_settings, mean0=np.zeros(p), cov0=np.eye(p),
                                               burnin=100)
        return c, States0, Rewards0, Actions0, int(cp_dict[last_trans[k]]) #, int(system_settings['changepoints'][-1])
    
    results = Parallel(n_jobs=-1)(delayed(simulate_and_store)(k, i, k * N_per_cluster + i) for k in range(K_true) for i in range(N_per_cluster))
    
    for c, States0, Rewards0, Actions0, changepoint in results:
        States[c, :, :] = States0
        Rewards[c, :] = Rewards0
        Actions[c, :] = Actions0
        changepoints_true[c] = changepoint
    
    # normalize state variables
    def transform(x):
        return (x - np.mean(x)) / np.std(x)
    
    
    for i in range(p):
        States[:,:,i] = transform(States[:,:,i])
    print('finish data generating')
    return States, Rewards, Actions, changepoints_true, g_index_true             
#%% estimate the value of the estimated policy
env = load_env()
States, Rewards, Actions, changepoints_true, g_index_true, system_indicator, system_settings_list, changepoint_list, last_trans = generate_data(env)
underlying_g_index=np.repeat(list(range(K_true)), N_per_cluster)
rbf_bw = 0.1
metric = 'kerneldist'
if type_est not in ["cluster_atend", "cluster_atend2"]:
    param_grid = {"max_depth": [3, 5],#
                  "min_samples_leaf": [50, 60, 70]}#
elif type_est == "cluster_atend":
    param_grid = {"max_depth": [3, 5],#
                  "min_samples_leaf": [5, 6, 7]}#
else:
    param_grid = {"max_depth": [2, 3],#
                  "min_samples_leaf": [3, 4, 5]}#
basemodel = DecisionTreeRegressor(random_state=seed)
change_points_subsequent = [changepoint_list, T_initial + np.random.poisson(true_cp_interval)]
change_point_next = change_points_subsequent[1]
while change_point_next < T_new - 30:
    change_point_next += np.random.poisson(true_cp_interval)
    change_points_subsequent.append(change_point_next)
change_points_subsequent = change_points_subsequent[:-1]
print("change_points =", change_points_subsequent)
change_points_subsequent.append(T_new)
print(list(range(T_initial, T_new, cp_detect_interval)))
print(list(range(T_initial, T_new, true_cp_interval)))
system_list = []

def estimate_value(States, Rewards, Actions, type_est, param_grid, basemodel, system_indicator):
    method = ''.join([i for i in type_est if not i.isdigit()])
    print('method', method)
    
    
    States_updated = copy(States)
    Rewards_updated = copy(Rewards)
    Actions_updated = copy(Actions)
    # number of data batches
    n_batch = int((T_new - T_initial) / cp_detect_interval)
    batch_seeds = np.random.randint(0,1e5, n_batch)
    cp_index = 0
    cp_current = np.repeat(0, N_per_cluster*K_true)
    T_current = T_initial
    T_length = T_initial
    change_or_not = {g: 0 for g in range(K_true)}
    print('system_indicator',system_indicator)
    for batch_index in range(n_batch):
        print('======= batch_index ======', batch_index)
        batch_data_name = "batch"+str(batch_index)+".dat"
        #%% if we have finished this batch
        if os.path.exists(batch_data_name) and not refit:
            print('batch_data_name', batch_data_name, 'exist!')
            sys.stdout.flush()
            
            with open(batch_data_name, 'rb') as f:
                batch_data = pickle.load(f)
            States_updated = batch_data["States_updated"]
            Rewards_updated = batch_data["Rewards_updated"]
            Actions_updated = batch_data['Actions_updated']
            T_current = batch_data['T_current']
            cp_index = batch_data['cp_index']
            cp_current = batch_data['cp_current']
            g_index = batch_data['g_index']
            change_or_not = batch_data['change_or_not']
            system_indicator = batch_data['system_indicator']
            print('batch data for', batch_index, 'loaded!', 'av_r', np.mean(Rewards_updated[:, T_initial:]))
            sys.stdout.flush()
            continue
        #%% first, detect change point
        cpTime = datetime.now()
        print('system_indicator',system_indicator)
        if method == "overall":
            print("cp_current =", cp_current)
            g_index = np.repeat(0, N_per_cluster*K_true)
        if method == "oracle":
            if batch_index == 0:
                cp_current = changepoints_true# np.minimum(changepoints_true, np.repeat(T_current, N))
                g_index = g_index_true.copy()
            else:
                for g in range(K_true):
                    if change_or_not[g] != 0:
                        cp_current[underlying_g_index == g] = np.repeat(change_points_subsequent[cp_index], N_per_cluster)
                if system_indicator[0] == system_indicator[1]:
                    if last_trans[system_indicator[0]] == last_trans[-1]:
                        g_index[:] = 0
                    else:
                        g_index[:N_per_cluster*2] = 0
                        g_index[N_per_cluster*2:] = 1
                else:
                    if last_trans[system_indicator[0]] == last_trans[-1]:
                        g_index[:N_per_cluster] = 0 
                        g_index[N_per_cluster:-N_per_cluster]=1
                        g_index[-N_per_cluster:] = 0
                    elif last_trans[system_indicator[1]] == last_trans[-1]:
                        g_index[:N_per_cluster] = 1 
                        g_index[N_per_cluster:]=0
                    else:
                        g_index[:N_per_cluster] = 1 
                        g_index[N_per_cluster:2*N_per_cluster]=0
        if method == "only_clusters":
            if batch_index != 0:
                T_length = States_updated.shape[1] 
            print('T_length', T_length, ', cp_current, ',cp_current)
            States_s = copy(States_updated)
            if is_transform:
                for i in range(p):
                    States_s[:, :, i] = transform(States_s[:, :, i])
            # tun K in iterations
            out = mean_detect.fit(States_s, Actions_updated,
                                    seed = seed+batch_index, 
                                    init = "changepoints", nthread=nthread, 
                                    changepoints_init =cp_current,
                                    Kl_fun=Kl_fun,
                                    max_iter=1,is_only_cluster=1, C=C, K=K_list)
            g_index = out[1]
            
            
        if method == "only_cp":
            g_index = np.repeat(0, N_per_cluster*K_true)
            if batch_index == 0:
                kappa_min = int(0.1*T_initial)
                kappa_max = int(0.9*T_initial)
                kappa_interval = int(T_initial/50)
            else:
                T_length = States_updated[:, int(np.max(cp_current)):, :].shape[1] #- 1
                kappa_min = int(fixed_kappamin*T_length) if fixed_kappamin <1 else min(T_length - 10, 50)
                kappa_max = min(int(0.9*T_length), 120)
                kappa_interval = max(int((kappa_max-kappa_min)/50), 1)
            States_s = copy(States_updated[:, int(np.max(cp_current)):, :])
            if is_transform:
                for i in range(p):
                    States_s[:, :, i] = transform(States_s[:, :, i])
            epsilon = 1/T_length
            print('!!!! is_cp_parallel', is_cp_parallel)
            out = mean_detect.fit_tuneK([1], States_s, Actions_updated[:, int(np.max(cp_current)):],
                                 seed = seed+batch_index, init = "clustering", epsilon=epsilon, nthread=nthread,
                                 kappa_min = kappa_min, kappa_max = kappa_max,
                                 kappa_interval=kappa_interval, max_iter=1, 
                                 g_index_init_list=[g_index], C=0, is_only_cp=1,
                                 threshold_type=threshold_type,
                                 is_tune_parallel=0, 
                                 is_cp_parallel = is_cp_parallel, save_path=path_name+'/threshold')
            cp_current += out.best_model[2]
            
            
        if method in ["cluster_atend","cluster_atend2"]:
            if batch_index != 0:
                T_length = States_updated.shape[1] 
                
            # print('T_length', T_length, ', cp_current, ',cp_current)
            States_s = copy(States_updated)
            if is_transform:
                for i in range(p):
                    States_s[:, :, i] = transform(States_s[:, :, i])
            cp_current = np.ones(N_per_cluster*K_true, dtype=int) * int(T_length-2)
            out = mean_detect.fit(States_s, Actions_updated,
                                    seed = seed+batch_index, 
                                    init = "changepoints", nthread=nthread, 
                                    changepoints_init =cp_current,
                                    Kl_fun=Kl_fun,
                                    max_iter=1,is_only_cluster=1, C=C, K=K_list)
            g_index = out[1]
            
            
        if method == "cp_indi":
            g_index = np.arange(N_per_cluster*K_true)
            if batch_index == 0:
                kappa_min = int(0.1*T_initial)
                kappa_max = int(0.9*T_initial)
                kappa_interval = int(T_initial/50)
            else:
                T_length = States_updated[:, int(np.max(cp_current)):, :].shape[1] #- 1
                kappa_min = int(fixed_kappamin*T_length) if fixed_kappamin <1 else min(T_length - 10, 50)
                kappa_max = min(int(0.9*T_length), 120)
                kappa_interval = max(int((kappa_max-kappa_min)/50), 1)
            States_s = copy(States_updated[:, int(np.max(cp_current)):, :])
            if is_transform:
                for i in range(p):
                    States_s[:, :, i] = transform(States_s[:, :, i])
            epsilon = 1/T_length
            print('!!!! is_cp_parallel', is_cp_parallel)
            out = mean_detect.fit_tuneK([int(N_per_cluster*K_true)], States_s, Actions_updated[:, int(np.max(cp_current)):],
                                 seed = seed+batch_index, init = "clustering", epsilon=epsilon, nthread=nthread,
                                 kappa_min = kappa_min, kappa_max = kappa_max,
                                 kappa_interval=kappa_interval, max_iter=1, 
                                 g_index_init_list=[g_index], C=0, is_only_cp=1,
                                 threshold_type=threshold_type,
                                 is_tune_parallel=0, is_cp_parallel = is_cp_parallel,
                                 save_path=path_name+'/threshold')
            cp_current += out.best_model[2]
            
        if method == "proposed":
            if batch_index == 0:
                cp_current = np.repeat(0,N_per_cluster*K_true)
                kappa_min = int(0.1*T_initial)
                kappa_max = int(0.9*T_initial)
                kappa_interval = int(T_initial/50)
                res = find_first_batch_proposed_cp()
                if res:
                    cp_current = np.repeat(int(np.max(cp_current)), N_per_cluster*K_true)+ res[0] #change_point_detected['integral_emp']
                    g_index = res[1]
                    need_fitting=0 
                else:
                    need_fitting=1
            else:
                need_fitting=1
                T_length = States_updated[:, int(np.max(cp_current)):, :].shape[1] #- 1
                kappa_min = int(fixed_kappamin*T_length) if fixed_kappamin <1 else 5
                kappa_max = int(0.9*T_length) # , 120min(T_length - 10, 50)
                kappa_interval = max(int((T_length)/50), 1)
            print('T_length', T_length,'kappa_max', kappa_max ,', cp_current, ',cp_current)
            States_s = copy(States_updated[:, int(np.max(cp_current)):, :])
            if is_transform:
                for i in range(p):
                    States_s[:, :, i] = transform(States_s[:, :, i])
            epsilon = 1/T_length
            print('States_s.shape', States_s.shape)
            if need_fitting:
                batch_proposed_file ="proposed_batch"+str(batch_index)+".dat"
                if os.path.exists(batch_proposed_file):
                    with open(batch_proposed_file, "rb") as f:
                        proposed_dat = pickle.load(f)
                        
                    cp_current = np.repeat(int(np.max(cp_current)), N_per_cluster*K_true)+ proposed_dat["changepoints"]
                    g_index = proposed_dat["g_index"]
                    
                else:
                    try:
                        out = mean_detect.fit(States_s, Actions_updated[:, int(np.max(cp_current)):],
                                            seed = seed+batch_index, init = "clustering", epsilon=epsilon,  nthread=nthread,
                                            kappa_min = kappa_min, kappa_max = kappa_max, 
                                            kappa_interval = kappa_interval,
                                            max_iter=max_iter, 
                                            K=K_list, init_cluster_range = T_length-1-kappa_min,
                                            threshold_type=threshold_type,
                                            Kl_fun=Kl_fun,  
                                            is_cp_parallel=is_cp_parallel, C=C, 
                                            early_stopping=early_stopping)
                        print('is_cp_parallel', is_tune_parallel, ', is_cp_parallel', is_cp_parallel, ', finish time: ',datetime.now()-startTime)
                        cp_current = np.repeat(int(np.max(cp_current)), N_per_cluster*K_true)+ out.changepoints #change_point_detected['integral_emp']
                        g_index = out.g_index
                        with open(batch_proposed_file, "wb") as f:
                            pickle.dump({"changepoints":out.changepoints,
                                         "g_index":out.g_index}, f)
                            
                        # print('system_indicator',system_indicator)
                    except:
                        print('!!! BUG in mean_detect')
                        with open('bugdata_proposed.dat' , "wb") as f:
                            pickle.dump({'States':States_s,
                                          'Actions':Actions_updated[:, int(np.max(cp_current)):]}, f)
                        exit()
                        with open('D:/OneDrive/PhD/DIRL/IHS/simu/simu_anonymous/semisyn_2020/online_value/bugdata_proposed.dat', 'rb') as f:
                            dat = pickle.load(f)
                        
                        States = dat["States"]
                        Actions = dat['Actions']
                        out = mean_detect.fit(States, Actions,
                                            seed = seed, init = "clustering", epsilon=epsilon,  nthread=nthread,
                                            kappa_min = kappa_min, kappa_max = kappa_max, 
                                            kappa_interval = kappa_interval,
                                            max_iter=max_iter, 
                                            K=K_list, init_cluster_range = T_length-1-kappa_min,
                                            threshold_type=threshold_type,
                                            Kl_fun=Kl_fun,  
                                            is_cp_parallel=is_cp_parallel, C=C, 
                                            early_stopping=early_stopping)
        
        
            print('Finished. Time: ', datetime.now() - cpTime)
            
        #%% estimate the optimal policy
        q_all_group = {g:None for g in set(g_index)}  
        q_all_fit = {g:None for g in set(g_index)}  
        print('States_updated.shape', States_updated.shape)
        print(cp_current, g_index)
        sys.stdout.flush()
        policyTime = datetime.now()
        for g in np.unique(g_index):
            print('g', g, ', batch_index, ',batch_index)
            print('cp_current', cp_current, cp_current[g_index == g][0] )
            print('States_updated[g_index == ',g,', ', cp_current[g_index == g][0], ':, :]',States_updated[g_index == g, cp_current[g_index == g][0]:, :].shape)
            
            policy_file = "batch_"+str(batch_index)+'_g'+str(g)+'.pkl'
            q_all_group[g] =stat.q_learning(States_updated[g_index == g, cp_current[g_index == g][0]:, :],
                                            Rewards_updated[g_index == g, cp_current[g_index == g][0]:].reshape((np.sum(g_index == g), -1)), 
                                            Actions_updated[g_index == g, cp_current[g_index == g][0]:], qmodel, degree, gamma,
                                            rbf_dim=degree, rbf_bw=rbf_bw)

            if os.path.exists(policy_file) and not refit:
                with open(policy_file, 'rb') as f:
                    q_tmp = pickle.load(f)
                q_all_group[g].q_function_list = q_tmp
                continue
            
            try:
                nthread_cv = 5 if nthread > 5 else nthread
                
                out = select_model_cv(States_updated[g_index == g, cp_current[g_index == g][0]:, :].reshape((np.sum(g_index == g), -1, p)), 
                                      Rewards_updated[g_index == g, cp_current[g_index == g][0]:].reshape((np.sum(g_index == g), -1)),
                                      Actions_updated[g_index == g, cp_current[g_index == g][0]:].reshape((np.sum(g_index == g), -1)),
                                      param_grid, bandwidth=rbf_bw,
                                      qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                                      nfold = 5, num_threads = nthread_cv, metric = metric,
                                      early_stopping=early_stopping, verbose=0)
                model = out['best_model']
                print(model)
                # q_all_group[g] =stat.q_learning(States_updated[g_index == g, cp_current[g_index == g][0]:, :],
                #                                 Rewards_updated[g_index == g, cp_current[g_index == g][0]:].reshape((np.sum(g_index == g), -1)), 
                #                                 Actions_updated[g_index == g, cp_current[g_index == g][0]:], qmodel, degree, gamma,
                #                                 rbf_dim=degree, rbf_bw=rbf_bw)
                q_all_fit[g] = q_all_group[g].fit(model, max_iter=200, tol = 1e-6,
                                                  early_stopping=early_stopping, verbose=1)
                sys.stdout.flush()
                
                with open(policy_file, 'wb') as f:
                    pickle.dump(q_all_fit[g].q_function_list, f)
                    
            except:
                print('!!! BUG in policy learning')
                with open('bugdatalearning.dat' , "wb") as f:
                    pickle.dump({'States':States_updated[g_index == g, cp_current[g_index == g][0]:, :].reshape((np.sum(g_index == g), -1, p)),
                                 'Actions':Actions_updated[g_index == g, cp_current[g_index == g][0]:].reshape((np.sum(g_index == g), -1)),
                                 'Rewards':Rewards_updated[g_index == g, cp_current[g_index == g][0]:].reshape((np.sum(g_index == g), -1))}, f)
                exit()
                ########## DEBUG
                with open('D:/OneDrive/PhD/DIRL/IHS/simu/simu_anonymous/semisyn_2020/online_value/bugdatalearning.dat', 'rb') as f:
                    dat = pickle.load(f)
                
                States = dat["States"]
                Actions = dat["Actions"]
                Rewards = dat["Rewards"]
                out = select_model_cv(dat["States"], 
                                      dat["Rewards"], dat["Actions"],
                                      param_grid, bandwidth=rbf_bw,
                                      qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                                      nfold = 5, num_threads = nthread, metric = metric,
                                      early_stopping=early_stopping, verbose=1)
                model = out['best_model']
                print(model)
                q_all_group[g] =stat.q_learning(dat["States"],
                                                dat["Rewards"], 
                                                dat["Actions"], qmodel, degree, gamma,
                                                rbf_dim=degree, rbf_bw=rbf_bw)
                q_all_fit[g] = q_all_group[g].fit(model, max_iter=200, tol = 1e-6,
                                                  early_stopping=early_stopping, verbose=0)

    
        print('Finished. Time: ', datetime.now() - policyTime)
        #%% now we collect the next batch of data following the estimated policy
        T_current += cp_detect_interval
        print("batch_index =", batch_index, "cp_index =", cp_index, "T_to_simulate =", T_current)
        # if the next change point has not been encountered, just keep the current dynamics
        print('change_points_subsequent[',cp_index+1,']', change_points_subsequent[cp_index+1])
        if T_current -1 <= change_points_subsequent[cp_index+1]:
            change_or_not = {g:0 for g in range(K_true)} # whether encountered a change point. change_or_not[0] is the indicator for true group 0 
        # if the next change point is encountered, need to make it the change point
        else:
            np.random.seed(batch_seeds[batch_index])
            change_or_not[0] = np.random.binomial(1, 0.5, 1).item()
            change_or_not[1] = np.random.binomial(1, 0.5, 1).item()
            for g in range(2):
                # if change, switch the generating system
                if change_or_not[g] > 0.5:
                    system_indicator[g] = np.abs(system_indicator[g]-1) 
            for g in range(2, K_true):
                change_or_not[g] = 0
            # if K_true>2:
                # change_or_not.extend([0 for _ in range(K_true-2)])
            print("change print encountered", change_or_not)
            cp_index += 1
        print('sys', system_indicator)
        
        # simulate new data following the estimated policy

        States_new = np.zeros([N_per_cluster*K_true, cp_detect_interval+1, p])
        Rewards_new = np.zeros([N_per_cluster*K_true, cp_detect_interval])
        Actions_new = np.zeros([N_per_cluster*K_true, cp_detect_interval])
        
        def simulate_and_store2(i):
            S0 = States_updated[i, -1, :]
            S0 = S0.reshape((1, 1, -1))
            g = underlying_g_index[i] #g_index_true[i]
            system_settings = system_settings_list[system_indicator[g]]
            if change_or_not[g] == 0:
                system_settings['changepoints'] = [0]
            else:
                system_settings['changepoints'] = [change_points_subsequent[cp_index] - T_current + cp_detect_interval]
            States0, Rewards0, Actions0 = simulate(system_settings, S0=S0, optimal_policy_model=q_all_group[g_index[i]]) #, cov=cov)
            return States0, Rewards0, Actions0, i  
        
   
        results = Parallel(n_jobs=-1)(delayed(simulate_and_store2)(i) for i in range(N_per_cluster*K_true))
        
        for States0, Rewards0, Actions0, i in results:
            States_new[i, :, :] = States0
            Rewards_new[i, :] = Rewards0
            Actions_new[i, :] = Actions0
            
        if is_r_orignial_scale:
            Rewards_new = transform_back(Rewards_new)
        sys.stdout.flush()
        States_updated = np.concatenate((States_updated, States_new[:,1:,:]), axis = 1)
        Rewards_updated = np.concatenate((Rewards_updated, Rewards_new), axis = 1)
        Actions_updated = np.concatenate((Actions_updated, Actions_new), axis = 1)
        
        
            
        print('States_updated.shape', States_updated.shape)
        print('rewards. ', np.mean(Rewards_updated[:, T_initial-1:]), 'epoch rewards',np.mean(Rewards_new))
        for g in set(g_index):
            mean_g = np.mean(Rewards_new[g_index==g,:])
            print(g ,'rewards', mean_g)
        system_list.append(system_indicator)
        sys.stdout.flush()
        
        
        with open(batch_data_name, "wb") as f:
            pickle.dump({
                'States_updated':States_updated,
                'Rewards_updated':Rewards_updated,
                'Actions_updated':Actions_updated,
                'T_current':T_current,
                'cp_index':cp_index,
                'cp_current':cp_current,
                'g_index':g_index,
                'change_or_not':change_or_not,
                'system_indicator':system_indicator
                }, f)

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
value, system_list = estimate_value(States, Rewards, Actions, type_est=type_est, param_grid=param_grid, basemodel=basemodel,
                                    system_indicator=system_indicator)
print('Finished. Time: ', datetime.now() - startTime)
print(type_est, "discounted reward:", value['discounted_reward'], "\n")
print(type_est, "average reward:", value['average_reward'], "\n")
with open(file_name , "wb") as f:
    pickle.dump({'value':value,
                 'system':system_list}, f)


sys.stdout.flush()


sys.stdout.close()
sys.stdout = stdoutOrigin
