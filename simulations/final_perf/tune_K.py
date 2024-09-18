# -*- coding: utf-8 -*-
"""
Test the performance of tuning K in each iteration
"""
#!/usr/bin/python
import platform, sys, os, pickle, re
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the "ClusterRL" directory
cluster_rl_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Append the ClusterRL directory to sys.path
sys.path.append(cluster_rl_dir)

import numpy as np
import functions.simulate_data_1d as sim
from datetime import datetime
import functions.simu_mean_detect as mean_detect
from sklearn.metrics.cluster import adjusted_rand_score
#%%
print('sys.argv', sys.argv)
seed = int(sys.argv[1])
init = sys.argv[2]
N = int(sys.argv[3])
T = int(sys.argv[4])
trans_setting = sys.argv[5]
nthread = int(sys.argv[6])
effect_size= sys.argv[7]
max_iter=int(sys.argv[8])
refit=int(sys.argv[9])
K_list =  [int(x) for x in sys.argv[10:]]
print(K_list)

Kl_fun='Nlog(NT)/T' 
C=1/5
# %% environment setup
# create folder under seed if not existing
def setpath(trans_setting, init = "true clustering"):
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
    if init == "tuneK_iter":
        init_name = init+ Kl_fun.replace("/", "_")+"_C"+str(C)+'/max_iter'+str(max_iter)
    else:
        init_name = init
    path_name = 'results/trans' + trans_setting +'/effect_size_' + effect_size+\
        '/N' + str(N) +'_T' + str(T)+'/K'+ str(K_list)+'/init_'+init_name+\
                 '/seed'+str(seed) 
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    os.chdir(path_name)
    
setpath(trans_setting, init=init)
print(os.getcwd())
sys.stdout.flush()
file_name = "seed_"+str(seed)+".dat"

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
    
else:
    need_fitting=True
    # exit()
        
# if os.path.exists(file_name):
#     exit()
    
# direct the screen output to a file
stdoutOrigin = sys.stdout
if refit:
    sys.stdout = open("log.txt", "w")
else:
    sys.stdout = open("log.txt", "a")
print("\nName of Python script:", sys.argv[0])
sys.stdout.flush()
#%%
np.random.seed(seed)
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
# width of smooth transition function
w = 0.01
delta = 0.05#1/T

# cluster number
# K_list = range(1, 5)

# parallel
epsilon = 1/T
B=10000
reward_setting = 'homo'

threshold_type = "maxcusum"
#%%
# [c1 *St+ c2*(2A-1) + c3*St*(2A-1)+c0]
if effect_size == "strong":
    effect_size_factor = 1.0
elif effect_size == "moderate":
    effect_size_factor = 0.5
elif effect_size == "weak":
    effect_size_factor = 0.2
else:
    effect_size_factor = int(effect_size)
coef =[[0,0,0,-0.5*effect_size_factor],[0,0,0,0.5*effect_size_factor]] # shift in mean
def gen_dat(N, T, K, coef, changepoint_list=None,trans_setting="pwconst2",
            seed=1):
    if changepoint_list is None:
        changepoint_list = [int(T/2)  + int(0.2 * T)- 1, int(T/2) - int(0.2 * T) - 1] # 如果这里为29， 那么第30个点是按照新的transition function计算的
    print('changepoint_list',changepoint_list)
    changepoints_true = np.zeros([N, 1])
    States = np.zeros([N, T, p])
    Rewards = np.zeros([N, T-1])
    Actions = np.zeros([N, T-1])
    for i in range(K):
        if i == 0:
            coef_tmp = coef
        else:
            coef_tmp = [coef[1],coef[0]]
        changepoint = changepoint_list[i]
        sim_dat = sim.simulate_data(int(N/K), T, changepoint, delta)
        if trans_setting == 'pwconst2' and reward_setting == 'homo':
            def mytransition_function(t):
                return sim_dat.transition_pwconstant2(t, mean, cov, coef=coef_tmp)
            def myreward_function(t):
                return sim_dat.reward_homo()
            States0, Rewards0, Actions0 = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed + changepoint_list.index(changepoint), T1=T-1)
        elif trans_setting == 'smooth' and reward_setting == 'homo':
            def mytransition_function(t):
                return sim_dat.transition_smooth2(t, mean, cov, coef=coef_tmp, w=w)
            def myreward_function(t):
                return sim_dat.reward_homo()
            States0, Rewards0, Actions0 = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed + changepoint_list.index(changepoint), T1=T-1)
        States[changepoint_list.index(changepoint)*int(N/K)+ 0:changepoint_list.index(changepoint)*int(N/K)+int(N/K), :, :] = States0
        Rewards[changepoint_list.index(changepoint)*int(N/K)+ 0:changepoint_list.index(changepoint)*int(N/K)+int(N/K), :] = Rewards0
        Actions[changepoint_list.index(changepoint)*int(N/K)+ 0:changepoint_list.index(changepoint)*int(N/K)+int(N/K), :] = Actions0
        changepoints_true[changepoint_list.index(changepoint)*int(N/2)+ 0:changepoint_list.index(changepoint)*int(N/2)+int(N/2), ] = changepoint_list[i] * np.ones([int(N/K),1])
    return States, Rewards, Actions, changepoints_true

# normalize state variables
def transform(x):
    return (x - np.mean(x)) / np.std(x)

States, Rewards, Actions, changepoints_true = gen_dat(N, T, 2, coef, 
                                                      None, trans_setting, seed)
for i in range(p):
    States[:,:,i] = transform(States[:,:,i])

g_index_true = np.repeat([0,1], [int(N/2),int(N/2)], axis=0)

kappa_min = int((T- 1 - np.max(changepoints_true))*0.8)
kappa_max = max(T-1, int((T- 1 - np.min(changepoints_true))*1.2))
# %% evaluation function
def evaluate(changepoints_true, g_index, predict, T):
    '''
    g_index : predicted group index
    predict : predicted changepoints
    '''
    changepoint_err = np.mean(np.abs(predict - changepoints_true)/T)
    cluster_err = adjusted_rand_score(changepoints_true, g_index)
    return changepoint_err, cluster_err

startTime = datetime.now()
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
if init == "tuneK_iter":
    out = mean_detect.fit(States, Actions, example="cdist", seed = seed,
                          init = "clustering",  B=B,
                          epsilon=epsilon,  nthread=nthread,threshold_type= threshold_type,
                          kappa_min = kappa_min, kappa_max = kappa_max,
                          K=K_list, init_cluster_range = T - 1 - kappa_min,
                          Kl_fun=Kl_fun, max_iter = max_iter,
                          C = C)
    g_index_pred = out.g_index
    cp_pred = out.changepoints
    loss = out.loss
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
                                g_index_pred.squeeze(), cp_pred.squeeze(), N, T)
print('changepoint_err',changepoint_err, 'ARI',ARI)
#%%
runtime = datetime.now() - startTime
print('Finished. Time: ',runtime)
print('path', os.getcwd())
if init == 'tuneK_iter':
    with open(file_name, "wb") as f:
        pickle.dump({'changepoints':cp_pred, 'clustering':g_index_pred, 
                     'cp_err':changepoint_err,'ARI':ARI,
                     'loss':loss,
                     'iter_num':out.iter_num,
                      'K_path':out.K_path
                     }, f)
else:
    with open(file_name, "wb") as f:
        pickle.dump({'changepoints':cp_pred, 'clustering':g_index_pred, 
                     'cp_err':changepoint_err,'ARI':ARI,
                     'loss':loss,
                     'iter_num':out[2].iter_num
                     }, f)