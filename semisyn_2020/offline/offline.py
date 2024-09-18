# -*- coding: utf-8 -*-
"""
Offline estimation performance

Created on Fri Jul 26 10:47:47 2024

@author: Liyuan
"""
import sys, os, pickle
# import dill as pickle
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the "DIRL" directory
dirl_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Append the ClusterRL directory to sys.path
sys.path.append(dirl_dir)

import numpy as np
# import matplotlib.pyplot as plt
from datetime import datetime
import functions.simu_mean_detect as mean_detect
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.cluster import adjusted_rand_score
from functions.simulate_data_1d_flexible import simulate
# import copy
from joblib import Parallel, delayed
path_name = os.getcwd()
#%%
print(sys.argv)
seed = int(sys.argv[1])
effect_size_factor = float(sys.argv[2])
init = sys.argv[3]
refit=int(sys.argv[4])
max_iter=int(sys.argv[5])
nthread=int(sys.argv[6])
T=int(sys.argv[7])
N_per_cluster=int(sys.argv[8])
trans_setting = sys.argv[9]
K_true = int(sys.argv[10])
stationary_team_dynamic=int(sys.argv[11])
kappa_interval_step=int(sys.argv[12])
env_type=sys.argv[13]
cov=float(sys.argv[14])
transform_s=int(sys.argv[15])
stationary_changeactionsign=int(sys.argv[16])
K_list =  [int(x) for x in sys.argv[17:]]

startTime = datetime.now()
np.random.seed(seed)
print("seed =", seed)

# %% simulate data
epsilon = 1/T
threshold_type = "maxcusum"
B = 10000
kappa_max=int(0.9*T)
kappa_min = int(0.1*T)
kappa_interval=int(T/kappa_interval_step)
# K_list = list(range(1,5))
Kl_fun="Nlog(NT)/T"
C=1
early_stopping=0 
ic_T_dynamic=0
p=3
reward_type= "next_state"
    
#%%
def setpath():
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
    if init in ["tuneK_iter", "only_clusters"]:
        init_name = init+ Kl_fun.replace("/", "_")+"_C"+str(C)+'/max_iter'+str(max_iter)+'_K_list'+str(K_list)
    else:
        init_name = init
    if transform_s!=0:
        init_name += '/statetrans'
    trans_setting_name = trans_setting + str(stationary_team_dynamic)
    if stationary_changeactionsign!=1 and stationary_team_dynamic<0:
        trans_setting_name+='_changesign'
    if env_type != "original":
        trans_setting_name +='/'+env_type
    path_name = 'results/trans' + trans_setting_name +'/effect_size_factor_' + str(effect_size_factor)+\
        '/Ncluster' + str(N_per_cluster) +'_T' + str(T)+'/K'+ str(K_true)+'/cov'+str(cov)+'/init_'+init_name+\
            '/kappa_max'+str(kappa_max)+'_min'+str(kappa_min)+'_step'+str(kappa_interval)+\
                 '/seed'+str(seed) 
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    os.chdir(path_name)
    


#%%
def load_env():
    dat_path = os.path.abspath(os.path.join(path_name, '..')) #_split_firstcluster
    if env_type == "original":
        env_path = dat_path +'/env/re_learnt_env.dat'
    elif env_type[:16] == 'randomstationary':
        env_path = dat_path +'/env/re_learnt_env_randomstationary.dat'
    else:
        env_path = dat_path +'/env/re_learnt_env_'+env_type+'.dat'
    print('env_path ',env_path )
    sys.stdout.flush()
    if os.path.exists(env_path):
        with open(env_path,'rb') as f:
            env = pickle.load(f)
    else:
        raise ValueError('Env not exist!')
    
    return env

def generate_data(env):
    def round_to_nearest(value):
        """Rounds a value to the nearest one in the given set possible_values."""
        if np.isscalar(value):
            value = np.array([value])
        possible_values = np.arange(kappa_min, min(kappa_max, T-1), step=kappa_interval, dtype=np.int32)
        idx = np.argmin(np.abs(possible_values - value), axis=-1)
        return possible_values[idx].reshape(value.shape)

    if env_type == "split_secondcluster":
        cp_cluster = [[int(T/2)], [round_to_nearest(int(T/3)).item(), round_to_nearest(int(2*T/3)).item()], [0]]

        transition_type = [[0, 1], [2, 3, 4], [stationary_team_dynamic, stationary_team_dynamic]]
    elif env_type == "original":
        cp_cluster = [[int(T/2)], [round_to_nearest(int(T/3)).item(), round_to_nearest(int(2*T/3)).item()], [0]]

        transition_type = [[0, 1], [0, 2, 3], [stationary_team_dynamic, stationary_team_dynamic]]
    elif env_type=='randomstationary':
        transition_type = [[0, 1], [2, 3], [stationary_team_dynamic, stationary_team_dynamic]]
        cp_cluster = [[int(T/2)], [int(T/2)], [0]]
    elif env_type in ["randomstationary_changesign", "randomstationary_changeactionsign"]:
        transition_type = [[-1, 1], [-3, 3], [stationary_team_dynamic, stationary_team_dynamic]]
        cp_cluster = [[int(T/2)], [int(T/2)], [0]]
    elif env_type in ["randomstationary3_changesign", "randomstationary3_changeactionsign"]:
        transition_type = [[-1, 1], [-3, 3], [stationary_team_dynamic, stationary_team_dynamic]]
        cp_cluster = [[int(T/2)], [int(T/2)], [0]]
        
    last_trans = [sublist[-1] for sublist in transition_type[:K_true]]
    g_index_true = np.repeat([list(set(last_trans)).index(i) for i in last_trans], N_per_cluster)
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
                    re_coef = -1*env["transition"][int(-1*g)]
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
    
    system_settings_list = [
        {'N': 1, 'T': T-1, 'p':p,
                           'changepoints': cp_cluster[k],
                           'state_functions':[transition_generator(env, transition_type[k][i]) for i in range(len(cp_cluster[k])+1)], # [transition_generator(env, transition_type[1][0]), transition_generator(env, transition_type[1][1])],
                           'reward_functions': [reward_generator(env, transition_type[k][i]) for i in range(len(cp_cluster[k])+1)], #[reward_generator(env, transition_type[1][0]), reward_generator(env, transition_type[1][1])],
                           'state_change_type': trans_setting,#,
                           'reward_change_type': trans_setting, #'homogeneous',
                           'delta':0.1
                            }
        for k in range(K_true)
        ]
        
    
    States = np.zeros([N_per_cluster*K_true, T, p])
    Rewards = np.zeros([N_per_cluster*K_true, T-1])
    Actions = np.zeros([N_per_cluster*K_true, T-1])
    changepoints_true = np.repeat(0, N_per_cluster*K_true)
    # g_index_true = np.repeat(list(range(K_true)), int(N_per_cluster))
    
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
    
    if transform_s:
        for i in range(p):
            States[:,:,i] = transform(States[:,:,i])
    print('finish data generating')
    return States, Rewards, Actions, changepoints_true, g_index_true

def find_online_fitted_file():
    # parameters
    fixed_kappamin=0.1
    is_transform=transform_s
    T_new = 401
    cp_detect_interval=25
    true_cp_interval=60
    # file_path
    if Kl_fun=='Nlog(NT)/T':
        Kl_fun_name = "Nlog(NT)_T"
    else:
        Kl_fun_name = Kl_fun
    if init == "tuneK_iter":
        type_est = "proposed"
    elif init in ["cluster_atend", "cp_indi"]:
        type_est = init
    else:
        type_est = init
    
    
    if type_est in ["proposed", "only_clusters",# "cluster_atend"
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
                                          # "cluster_atend", "cp_indi"
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
    online_dir = os.path.abspath(os.path.join(path_name, '..', 'online_value'))

    data_path = online_dir+'/results/trans' + trans_setting_name +'/N' + str(N_per_cluster) +'/Tnew_' +\
                str(T_new)+'_type_'+method_name+'_cpitv'+ str(cp_detect_interval)+'/true_cp'+str(true_cp_interval)+\
                '/cov'+str(cov) + '/seed'+str(seed)+"/batch0.dat"
                
                
    # method_name = method + str(K_list)+'/Kl_fun'+Kl_fun_name+'_C'+str(C)
    # method_name += "earlystop"+str(1)+'/max_iter'+str(max_iter)
    # if fixed_kappamin < 1:
    #     method_name += '/kappamin'+str(fixed_kappamin)
    
    # if is_transform != 1:
    #     method_name +='/transform'+str(is_transform)
        
    # if effect_size_factor != 1.0:
    #     trans_setting_name =trans_setting + "/effect_size"+str(effect_size_factor)
    # else:
    #     trans_setting_name = trans_setting
    # if env_type != "randomstationary":
    #     trans_setting_name += '/'+env_type
    # if reward_type != "next_state":
    #     trans_setting_name += '/reward'+reward_type
    # if K_true!=3:
    #     trans_setting_name+='K_true'+str(K_true)
    # if stationary_team_dynamic!=4 and K_true>2:
    #     trans_setting_name+= '/stationary'+str(stationary_team_dynamic)
    #     if stationary_team_dynamic<1 and stationary_changeactionsign!=1:
    #         trans_setting_name+= "changesign"
    
    # online_dir = os.path.abspath(os.path.join(path_name, '..', 'online_value'))

    # data_path = online_dir+'/results/trans' + trans_setting_name +'/N' + str(N_per_cluster) +'/Tnew_' +\
    #             str(T_new)+'_type_'+method_name+'_cpitv'+ str(cp_detect_interval)+'/true_cp'+str(true_cp_interval)+\
    #             '/cov'+str(cov) + '/seed'+str(seed) +"/proposed_batch0.dat"
    return data_path
# %% evaluation function
def evaluate(changepoints_true, g_index_true, g_index, predict, T):
    '''
    g_index : predicted group index
    predict : predicted changepoints
    '''
    changepoint_err = np.mean(np.abs(predict - changepoints_true)/changepoints_true) #np.mean(np.abs(predict - changepoints_true)/T)
    cluster_err = adjusted_rand_score(g_index_true, g_index)
    return changepoint_err, cluster_err


#%%
def main():
    _ = setpath()
    print(os.getcwd())
    sys.stdout.flush()
    file_name="seed"+str(seed)+".dat"
    
    if os.path.exists(file_name) and not refit:
        try:
            with open(file_name, 'rb') as f:
                dat = pickle.load(f)
            cp_pred = dat["changepoints"]
            g_index_pred = dat["clustering"]
            iter_num=dat["iter_num"]
            K_path=dat["K_path"]
            loss = dat["loss"]
            need_fitting=False 
        except EOFError:
            print(f"Error: The file {file_name} is either empty or corrupted.")
            # Handle the error accordingly, such as by setting `dat` to None or another fallback
            # dat = None
            need_fitting=True
        
        # with open(file_name, 'rb') as f:
        #     dat = pickle.load(f)
        # cp_pred = dat["changepoints"]
        # g_index_pred = dat["clustering"]
        # iter_num=dat["iter_num"]
        # K_path=dat["K_path"]
        # loss = dat["loss"]
        # need_fitting=False 
    else:
        need_fitting=True
        # exit()
        
    # direct the screen output to a file
    stdoutOrigin = sys.stdout
    if not refit:
        sys.stdout = open("log.txt", "a")
    else:
        sys.stdout = open("log.txt", "w")
    print("\nName of Python script:", sys.argv[0])
    sys.stdout.flush()
    
    #%% generate data
    env = load_env()
    States, Rewards, Actions, changepoints_true, g_index_true = generate_data(env) 
    print('changepoints_true',changepoints_true)
    print('g_index_true',g_index_true)
    #%%
    if init == "only_clusters" and need_fitting:
        # States, Rewards, Actions, changepoints_true, g_index_true = generate_data(env) 
        # print('changepoints_true',changepoints_true)
        # print('g_index_true',g_index_true)
        # kappa_max=int(0.9*T)
        # kappa_min=int(0.1*T)
        # kappa_interval=int(T/kappa_interval_step)
        online_file = find_online_fitted_file()
        print("online", online_file)
        if os.path.exists(online_file):
            print('found!')
            with open(online_file, "rb") as f:
                res= pickle.load(f)
            g_index_pred = res["g_index"]
            cp_pred = res["cp_current"]
            iter_num=max_iter
            K_path=[0]
            loss=0
        else:
            changepoints = changepoints_true # np.repeat(np.max(changepoints_true), N_per_cluster*K_true)
            changepoints  = np.zeros(N_per_cluster*K_true)
            out = mean_detect.fit(States, Actions, init = "changepoints",
                                  nthread=nthread, changepoints_init = changepoints, 
                                  max_iter=1, is_only_cluster=1, C=C, 
                                  K = K_list,
                                  Kl_fun=Kl_fun, ic_T_dynamic=ic_T_dynamic)
            g_index_pred = out.g_index
            cp_pred = out.changepoints
            print(g_index_pred)
            loss = out.loss
            iter_num=out.iter_num
            K_path=out.K_path
    #     changepoint_err, ARI = evaluate(changepoints_true.squeeze(),
    #                                     g_index_true,
    #                                     g_index_pred.squeeze(), cp_pred.squeeze(), T)
    #     print('changepoint_err',changepoint_err, 'ARI',ARI)
    # %%
    if init == "only_cp" and need_fitting:
        # States, Rewards, Actions, changepoints_true, g_index_true = generate_data(env) 
        # print('changepoints_true',changepoints_true)
        # print('g_index_true',g_index_true)
        
        # g_index = g_index_true
        # kappa_max=int(0.9*T)
        # kappa_min = int(0.1*T)
        # kappa_interval=int(T/kappa_interval_step)
        online_file = find_online_fitted_file()
        print("online", online_file)
        if os.path.exists(online_file):
            print('found!')
            with open(online_file, "rb") as f:
                res= pickle.load(f)
            g_index_pred = res["g_index"]
            cp_pred = res["cp_current"]
            iter_num=max_iter
            K_path=[0]
            loss=0
        else:
            g_index=np.zeros((0, N_per_cluster*K_true))
            out = mean_detect.fit(States, Actions, example="cdist", seed = seed,
                                  init = "clustering", B=B,
                                  epsilon=epsilon, nthread=nthread,
                                  kappa_min = kappa_min, kappa_max = kappa_max,
                                  max_iter=1, alpha=0.01,
                                  g_index_init=g_index,
                                  is_only_cp=1,
                                  K=K_list, init_cluster_range = T - 1 - kappa_min,
                                  Kl_fun=Kl_fun, C=C, threshold_type = threshold_type)
            g_index_pred = out.g_index
            cp_pred = out.changepoints
            print(g_index_pred)
            loss = out.loss
            K_path=out.K_path
            iter_num=out.iter_num
        # changepoint_err, ARI = evaluate(changepoints_true.squeeze(),
        #                                 g_index_true,
        #                                 g_index_pred.squeeze(), cp_pred.squeeze(), T)
        # print('changepoint_err',changepoint_err, 'ARI',ARI)
    #%% 1. separately run the whole algorithm with a fixed K from K_list using fit_tuneK()
    if init == "kmeans":
        out = mean_detect.fit_tuneK(K_list, States, Actions, example="cdist", seed = seed,
                              init = "clustering",  B=B,
                              epsilon=epsilon,  nthread=nthread,threshold_type= threshold_type,
                              kappa_min = kappa_min, kappa_max = kappa_max,
                              is_tune_parallel =0,
                              init_cluster_range = T - 1 - kappa_min)
        g_index_pred = out[2].g_index
        cp_pred= out[2].changepoints
        loss = out[2].loss
    #%% 2. tune K from K_list in each iteration using fit()
    if init == "tuneK_iter" and need_fitting:
        online_file = find_online_fitted_file()
        print("online", online_file)
        if os.path.exists(online_file):
            print('found!')
            with open(online_file, "rb") as f:
                res= pickle.load(f)
            g_index_pred = res["g_index"]
            cp_pred = res["changepoints"]
            iter_num=max_iter
            K_path=[0]
            loss=0
        else:
            out = mean_detect.fit(States, Actions, example="cdist", seed = seed,
                                  init = "clustering",  B=B,
                                  epsilon=epsilon,  nthread=nthread,threshold_type= threshold_type,
                                  kappa_min = kappa_min, kappa_max = kappa_max,
                                  kappa_interval=kappa_interval,
                                  K=K_list, init_cluster_range = T - 1 - kappa_min,
                                  Kl_fun=Kl_fun, max_iter = max_iter,
                                  C = C)
            g_index_pred = out.g_index
            cp_pred = out.changepoints
            loss = out.loss
            iter_num=out.iter_num
            K_path=out.K_path
            print('out.K_path', out.K_path)
        
    #%%
    if init == "cluster_atend2" and need_fitting:
        online_file = find_online_fitted_file()
        print("online", online_file)
        if os.path.exists(online_file):
            print('found!')
            with open(online_file, "rb") as f:
                res= pickle.load(f)
            g_index_pred = res["g_index"]
            cp_pred = res["cp_current"]
            iter_num=max_iter
            K_path=[0]
            loss=0
        else:
            cp_current = np.ones(States.shape[0]) * int(T-2)
            out = mean_detect.fit(States, Actions,
                                    seed = seed, 
                                    init = "changepoints", nthread=nthread, 
                                    threshold_type= threshold_type,
                                    changepoints_init =cp_current,
                                    kappa_min = kappa_min, kappa_max = kappa_max,
                                    kappa_interval=kappa_interval,
                                    Kl_fun=Kl_fun,
                                    max_iter=1,is_only_cluster=1, C=C, K=K_list)
            
            # out = mean_detect.fit(States, Actions, example="cdist", seed = seed,
            #                       init = "clustering",  B=B,
            #                       epsilon=epsilon,  nthread=nthread,threshold_type= threshold_type,
            #                       kappa_min = kappa_min, kappa_max = kappa_max,
            #                       kappa_interval=kappa_interval,
            #                       K=K_list, init_cluster_range = T - 1 - kappa_min,
            #                       Kl_fun=Kl_fun, max_iter = max_iter,
            #                       C = C)
            g_index_pred = out.g_index
            cp_pred = out.changepoints
            loss = out.loss
            iter_num=out.iter_num
            K_path=out.K_path
            print('out.K_path', out.K_path)
    #%%
    if init == "cp_indi":
        online_file = find_online_fitted_file()
        print("online", online_file)
        if os.path.exists(online_file):
            print('found!')
            with open(online_file, "rb") as f:
                res= pickle.load(f)
            g_index_pred = res["g_index"]
            cp_pred = res["cp_current"]
            iter_num=max_iter
            K_path=[0]
            loss=0
        else:
            g_index = np.arange(N_per_cluster*K_true)
            out = mean_detect.fit_tuneK([int(N_per_cluster*K_true)], States, 
                                        Actions,
                                        g_index_init_list=[g_index],
                                 seed = seed, init = "clustering", B=B,
                                 epsilon=epsilon,  nthread=nthread,threshold_type= threshold_type,
                                 kappa_min = kappa_min, kappa_max = kappa_max,
                                 kappa_interval=kappa_interval,
                                 K=K_list, init_cluster_range = T - 1 - kappa_min,
                                 Kl_fun=Kl_fun, max_iter = max_iter,
                                 C = C)
            
            g_index_pred = out.g_index
            cp_pred = out.changepoints
            loss = out.loss
            iter_num=out.iter_num
            K_path=out.K_path
            print('out.K_path', out.K_path)
            
    #%% tuning K after convergence for each K
    # if init == "tuneK_end":
    #     out = mean_detect.fit_tuneK(States, Actions, example="cdist", seed = seed,
    #                           init = "clustering",  B=B,
    #                           epsilon=epsilon,  nthread=nthread,threshold_type= threshold_type,
    #                           kappa_min = kappa_min, kappa_max = kappa_max,
    #                           K=K_list, init_cluster_range = T - 1 - kappa_min,
    #                           C = np.log(N*T))
    #     g_index_pred = out.g_index
    #     cp_pred = out.changepoints
    #     loss = out.loss
    #     print('out.K_path', out.K_path)
    #%% 3. fix K = 2
    if init == "K2":
        out =mean_detect.fit(States, Actions, example="cdist", seed = seed,
                              init = "clustering",  B=B,
                              epsilon=epsilon,  nthread=nthread,threshold_type= threshold_type,
                              kappa_min = kappa_min, kappa_max = kappa_max,
                              max_iter=max_iter,
                              K=2, init_cluster_range = T - 1 - kappa_min)
        g_index_pred = out.g_index
        cp_pred= out.changepoints
        loss = out.loss
    #%%
    changepoint_err, ARI = evaluate(changepoints_true.squeeze(),
                                    g_index_true,
                                    g_index_pred.squeeze(), cp_pred.squeeze(), T)
    print('cp_pred', cp_pred)
    print('g_index_pred',g_index_pred)
    print('changepoint_err',changepoint_err, 'ARI',ARI)
    #%%
    runtime = datetime.now() - startTime
    print('Finished. Time: ',runtime)
    print('path', os.getcwd())
    if init in ['tuneK_iter', "only_clusters", "only_cp", "cluster_atend", "cluster_atend2", "cp_indi"]:
        with open(file_name, "wb") as f:
            pickle.dump({'changepoints':cp_pred, 'clustering':g_index_pred, 
                         'cp_err':changepoint_err,'ARI':ARI,
                         'loss':loss,
                         'iter_num': iter_num,
                          'K_path': K_path
                         }, f)
    else:
        with open(file_name, "wb") as f:
            pickle.dump({'changepoints':cp_pred, 'clustering':g_index_pred, 
                         'cp_err':changepoint_err,'ARI':ARI,
                         'loss':loss,
                         'iter_num':out[2].iter_num
                         }, f)
#%%
if __name__ == '__main__':
    main()


