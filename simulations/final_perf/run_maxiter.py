# -*- coding:utf-8 -*-
#!/usr/bin/python
import platform, sys, os, pickle, re
sys.path.append('~/.local/lib64/python3.6/site-packages')
sys.path.append('/usr/lib64/python3.6/site-packages') 
import numpy as np
sys.path.append('/home/huly0209_gmail_com/heterRL')
import functions.simulate_data_1d as sim
from datetime import datetime
import functions.simu_mean_detect as mean_detect
from sklearn.metrics.cluster import adjusted_rand_score
import random

gamma = 0.9
print('sys.argv',sys.argv)
seed = int(sys.argv[1])
print('seed')
init = sys.argv[2]
N = int(sys.argv[3])
T = int(sys.argv[4])
trans_setting = sys.argv[5]
nthread = int(sys.argv[6])
cov = float(sys.argv[7])
threshold_type = sys.argv[8]
K = int(sys.argv[9])
max_iter = int(sys.argv[10])
random_cp_index = int(sys.argv[11])
reward_setting = 'homo'
print('init',init,'N',N)
B=2000

startTime = datetime.now()
# %% environment setup
# create folder under seed if not existing

def setpath(trans_setting, init = "true clustering"):
    #os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu/toyexample/error")
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)           
    path_name = './strong_threshold_'+ threshold_type+'/sim_result_trans' + trans_setting +'/N' + str(N) +'/T_' + str(T)+'_init_'+init+\
                '_1d/cov'+str(cov) + '/seed'+str(seed)   
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    os.chdir(path_name)
    
setpath(trans_setting, init=init)
if init == "random_cp":
    file_name = "seed_"+str(seed)+"_ind"+str(random_cp_index)+".dat"
else:
    file_name = "seed_"+str(seed)+".dat"
if os.path.exists(file_name):# and init != "random_cp" and init != "kmeans":
    exit()
    
# direct the screen output to a file
stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "w")
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
# cov = 1

# width of smooth transition function
w = 0.01
delta = 0.05#1/T

# cluster number
# K = 2

# parallel
nthread=3

epsilon = 1/T

effect_size = "strong"
if effect_size == "strong":
    effect_size_factor = 1.0
elif effect_size == "moderate":
    effect_size_factor = 0.5
elif effect_size == "weak":
    effect_size_factor = 0.2
else:
    effect_size_factor = int(effect_size)
coef =[[0,0,0,-0.5*effect_size_factor],[0,0,0,0.5*effect_size_factor]] # shift in mean
def gen_dat(N, T, K, coef, changepoint_list=None,trans_setting="pwsonst2",
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
for i in range(1):
    States[:,:,i] = transform(States[:,:,i])

g_index_true = np.repeat([0,1], [int(N/2),int(N/2)], axis=0)

kappa_min = int((T- 1 - np.max(changepoints_true))*0.8)
kappa_max = max(T-1, int((T- 1 - np.min(changepoints_true))*1.2))
# %% evaluation function
def evaluate(changepoints_true, g_index, predict, N, T):
    '''
    g_index : predicted group index
    predict : predicted changepoints
    '''
    changepoint_err = np.mean(np.abs(predict - changepoints_true)/T)
    cluster_err = adjusted_rand_score(changepoints_true, g_index)
    return changepoint_err, cluster_err

#%% 1. init with true clustering
if init == "true_clustering":
    out = mean_detect.fit(States, Actions, example="cdist", seed = seed,
                           init = "clustering", g_index_init= g_index_true, B=B,
                          epsilon=epsilon,  nthread=nthread, threshold_type= threshold_type,
                          kappa_min = kappa_min, kappa_max = kappa_max,
                          max_iter=max_iter)
    # changepoint_err, ARI = evaluate(changepoints_true.squeeze(),
    #                                         out[1].squeeze(), out[2].squeeze(), N, T)
    # print('changepoint_err',changepoint_err, 'ARI',ARI)

#%% 2. init with the true change point
if init == "true_change_points":
    out = mean_detect.fit(States, Actions, example="cdist", seed = seed,
                           init = "changepoints", changepoints_init = changepoints_true, B=B,
                          epsilon=epsilon, nthread=nthread,threshold_type= threshold_type,
                          kappa_min = kappa_min, kappa_max = kappa_max,
                          max_iter=max_iter)
    # changepoint_err, ARI = evaluate(changepoints_true.squeeze(),
    #                                         out[1].squeeze(), out[2].squeeze(), N, T)
    # print('changepoint_err',changepoint_err, 'ARI',ARI)

#%% 3. init with no change points
if init == "no_change_points":
    changepoints_init = np.zeros(changepoints_true.shape)
    out = mean_detect.fit(States, Actions, example="cdist", seed = seed, B=B,
                          init = "changepoints", changepoints_init = changepoints_init,
                          epsilon=epsilon, nthread=nthread,threshold_type= threshold_type,
                          kappa_min = kappa_min, kappa_max = kappa_max,
                          max_iter=max_iter)

#%% 4. no clusters
if init == "no_clusters":
    out = mean_detect.fit(States, Actions, example="cdist", seed = seed, B=B,
                          init = "changepoints", changepoints_init = None,threshold_type= threshold_type,
                          epsilon=epsilon, nthread=nthread, clustering_warm_start =0,
                          kappa_min = kappa_min, kappa_max = kappa_max,
                          max_iter=max_iter)
#%% 5. Randomly choose a time point
if init == "random_cp":
    changepoints_init = np.repeat(random.randint(0, T-2), N)
    out = mean_detect.fit(States, Actions, example="cdist", seed = seed+random_cp_index, B=B,
                          init = "changepoints", changepoints_init = changepoints_init,
                         epsilon=epsilon, nthread=nthread,threshold_type= threshold_type,
                         kappa_min = kappa_min, kappa_max = kappa_max,
                         K=K, max_iter=max_iter)
#%% 6. Random clustering
if init == "random_clustering":
    g_index_init = np.random.choice(K, N)
    out = mean_detect.fit(States, Actions, example="cdist", seed = seed, B=B,
                          init = "clustering", g_index_init = g_index_init,
                          epsilon=epsilon,nthread=nthread,threshold_type= threshold_type,
                          kappa_min = kappa_min, kappa_max = kappa_max,
                          max_iter=max_iter)
#%% 7. init with kmeans on the last time points
if init == "kmeans":
    out = mean_detect.fit(States, Actions, example="cdist", seed = seed,
                          init = "clustering",  B=B,
                          epsilon=epsilon,  nthread=nthread,threshold_type= threshold_type,
                          kappa_min = kappa_min, kappa_max = kappa_max,
                          K=K, max_iter=max_iter, init_cluster_range = T - 1 - kappa_min)
   
#%% 8. tune K in each iteration
if init == "tuneK_iter":
    K_list = range(1, 5)
    out = mean_detect.fit(States, Actions, example="cdist", seed = seed,
                          init = "clustering",  B=B,
                          epsilon=epsilon,  nthread=nthread,threshold_type= threshold_type,
                          kappa_min = kappa_min, kappa_max = kappa_max,
                          K=K_list, max_iter=max_iter, init_cluster_range = T - 1 - kappa_min)

#%%
changepoint_err, ARI = evaluate(changepoints_true.squeeze(),
                                out[1].squeeze(), out[2].squeeze(), N, T)
print('changepoint_err',changepoint_err, 'ARI',ARI)
#%%
print('Finished. Time: ', datetime.now() - startTime)
print('path', os.getcwd())
# file_name = "seed_"+str(seed)+".dat"
with open(file_name, "wb") as f:
    pickle.dump({'changepoints':out.changepoints, 'clustering':out.g_index, 
                 'cp_err':changepoint_err,'ARI':ARI,
                 'loss':out.loss}, f)