'''
Simulate 4->3 clusterings 2-dimensional time series data

'''
#!/usr/bin/python
import platform, sys, os, pickle, re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu/results")
# sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu/simu") # 引用模块的地址
# import functions.compute_test_statistics as stat
# import functions.simulate_data_pd as sim
# import functions.simu_mean_detect as mean_detect
# import functions.utilities as uti
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") # 引用模块的地址
import simu.simulate_data_pd as sim
import simu.compute_test_statistics as stat
import simu.simu_mean_detect as mean_detect
import simu.utilities as uti
from sklearn.metrics.cluster import adjusted_rand_score
import pprint
import pandas as pd

'''
Arguments passed:
- seed: int. random seed to generate data
- kappa: int. test whether data on [T-kappa, T] is stationary
- num_threads: int. number of threads to compute the test statistic over many u's in parallel
- gamma: float. discount factor for the cumulative discounted reward. between 0 and 1
- trans_setting: string. scenario of the transition function. 
    Takes value from 'homo' (homogeneous), 'pwconst2' (piece-wise constant), or 'smooth' (smooth)
- reward_setting: string. scenario of the reward function. 
    Takes value from 'homo' (homogeneous), 'pwconst2' (piece-wise constant), or 'smooth' (smooth)
- N: int. number of individuals
- RBFSampler_random_state: int. random seed to supply to RBFSampler function

Example:
seed = 29
kappa = 55
num_threads = 5
gamma = 0.9
trans_setting = 'smooth'
reward_setting = 'homo'
N = int(25)
RBFSampler_random_state = 1
'''
# Arguments passed
# seed = int(sys.argv[1])
# kappa = int(sys.argv[2])
# num_threads = int(sys.argv[3])
# gamma = float(sys.argv[4])
# trans_setting = sys.argv[5]
# reward_setting = sys.argv[6]
# N = int(sys.argv[7])
# RBFSampler_random_state = int(sys.argv[8])
seed = 1
gamma = 0.9
trans_setting = 'pwconst2'
reward_setting = 'homo'
N = int(3*12)
df=None
alpha=0.05
epsilon=0.05
C1=1
C2=0.5
np.random.seed(seed)
# %% simulate data
# terminal timestamp
T = 100
kappa = T
epsilon=0.05
# dimension of X0
p = 2
# mean vector of X0
mean0 = 0
# diagonal covariance of X0
cov0 = 0.25
# mean vector of random errors zt
mean = 0
# diagonal covariance of random errors zt
cov = 0.25

# width of smooth transition function
w = 0.01
delta = 1/10

# heter changpoints and means
K = 3

# parallel
nthread=3

signal = [[0.1, -0.1], [0.1, 0, -0.1]]

coef =[[[-0.1, 0, 0.25],[0.1, 0.4, 0.25],[-0.2, 0, 0.5],[-0.1, 0.25, 0.75]], 
        [[0, 0.6, 0.75],[-0.1, -0.4, 0.75], [0.35, 0.125, 0.625]]] # this is acceptable
coef =[[[-0.1, 0, 0.25],[0.1, 0.4, 0.25],[-0.2, 0, 0.5],[-0.1, 0.25, 0.75]], 
        [[-0.1, -0.4, 0.75],[0, 0.6, 0.75], [0.35, 0.125, 0.625]]] # this is acceptable 0609
# coef =[[[-0.1, 0, 0.25],[0.1, 0.4, 0.25],[-0.2, 0, 0.5],[-0.1, 0.25, 0.75]], 
#         [[0, 0.6, 0.75], [-0.1, -0.4, 0.75], [0.35, 0.125, 0.625]]] # shift in mean

changepoint_list = [29, 49, 69] 
def gen_dat(N, T, K, coef, signal, changepoint_list=None,
            trans_setting="pwsonst2", seed=1):
    np.random.seed(seed)
    if changepoint_list is None:
        changepoint_list = [int(T/2) +30 + int(0.1 * T) - 1, int(T/2)-1 +30, int(T/2) - int(0.1 * T) +30- 1] 
        # changepoint_list = [9, 49, 89] 
        # changepoint_list = [19, 49, 79] 

    changepoints_true = np.zeros([N, 1])
    States = np.zeros([N, T, p])
    Rewards = np.zeros([N, T-1])
    Actions = np.zeros([N, T-1])
    def myreward_function(t):
        return sim_dat.reward_homo()
    coef_tmp = [None] * 2
    changepoint = 0
    for i in range(N):
        if i < int(N/4):
            changepoint = changepoint_list[0]
            coef_tmp[0] = coef[0][0]
            coef_tmp[1] = coef[1][0]
            signal_tmp = [signal[0][0], signal[1][0]]
            # print('signal_tmp',signal_tmp)
        elif i < int(N/3):
            changepoint = changepoint_list[0]
            coef_tmp[0] = coef[0][1]
            coef_tmp[1] = coef[1][0]
            signal_tmp = [signal[0][0], signal[1][0]]
        elif i < int(N/2):
            changepoint = changepoint_list[1]
            coef_tmp[0] = coef[0][1]
            coef_tmp[1] = coef[1][1]
            signal_tmp = [signal[0][0], signal[1][1]]
        elif i < int(2*N/3):
            changepoint = changepoint_list[1]
            coef_tmp[0] = coef[0][2]
            coef_tmp[1] = coef[1][1]
            signal_tmp = [signal[0][1], signal[1][1]]
        elif i < int(3*N/4):
            changepoint = changepoint_list[2]
            coef_tmp[0] = coef[0][2]
            coef_tmp[1] = coef[1][2]
            signal_tmp = [signal[0][1], signal[1][2]]
        else:
            changepoint = changepoint_list[2]
            coef_tmp[0] = coef[0][3]
            coef_tmp[1] = coef[1][2]
            signal_tmp = [signal[0][1], signal[1][2]]
            
        sim_dat = sim.simulate_data(1, T, p, changepoint, delta)
        def mytransition_function(t):
            return sim_dat.transition_pwconstant2(t, mean, cov, coef_tmp,signal_tmp)
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
    g_index_true = np.append([np.zeros(int(N/3)), np.ones(int(N/3))], 2*np.ones(int(N/3)))
    Actions = Actions.astype(int)
    return States, Rewards, Actions, changepoints_true, g_index_true


#%% plot trajectory
# fig, ax = plt.subplots()
C1 = 4
C2 = 3
def plot_fun(dim = 0):
    plt.figure()
    plt.plot(States[::3,:,dim].T)
    plt.vlines([69,79,89], -4,4, linestyles='dashed', colors='r')
    plt.show()
    
    plt.figure()
    plt.plot(States[:int(N/(C1)):int(N/(C1*2)),:,dim].T, 'y', label="i="+str(0)+","+str(int(N/12)))
    plt.plot(States[(int(N/C1)):(int(N/C2)-1):int((int(N/C2) - int(N/C1))/2),:,dim].T, 'b--', label="i="+str(int(N/C1))+","+str(int(N/C1) + int((int(N/C2) - int(N/C1))/2)))
    plt.vlines(89, -3,3, linestyles='dashed', colors='r')
    plt.title("Group 1 1 & Group 12, merge")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    
    plt.figure()
    plt.plot(States[int(N/C1):(int(N/C2)-1):int((int(N/C2) - int(N/C1))/2),:,dim].T, 'y', label="i="+str(int(N/C1))+","+str(int(N/C1) + int((int(N/C2) - int(N/C1))/2)))
    plt.plot(States[(int(N/C2)):int(2*N/C1):int((int(2*N/C1)-int(N/C2))/2),:,dim].T, 'b--', label="i="+str((int(N/C2)))+","+str((int(N/C2))+int((int(2*N/C1)-int(N/C2))/2)))
    plt.vlines(79, -3,3, linestyles='dashed', colors='r')
    plt.title("Group 1 2, split")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

    plt.figure()
    plt.plot(States[int(2*N/C1):(int(2*N/C2)):int(((int(2*N/C2))-int(2*N/C1))/2),:,dim].T, 'y', label="i="+str(int(2*N/C1))+","+str(int(2*N/C1)+ int(((int(2*N/C2))-int(2*N/C1))/2)))
    plt.plot(States[(int(2*N/C2)):int(3*N/C1):round((int(3*N/C1)-int(2*N/C2))/2),:,dim].T, 'b--', label="i="+str((int(2*N/C2)))+","+str((int(2*N/C2))+round((int(3*N/C1)-int(2*N/C2))/2)))
    plt.vlines(69, -3,3, linestyles='dashed', colors='r')
    plt.title("Group 1 3 split")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    
    plt.figure()
    plt.plot(States[int(2*N/C2):int(3*N/C1):int((int(3*N/C1)-int(2*N/C2))/2),:,dim].T, 'y', label="i="+str((int(2*N/C2)))+","+str((int(2*N/C2))+round((int(3*N/C1)-int(2*N/C2))/2)))
    plt.plot(States[(int(3*N/C1)):(int(4*N/C1)-1):round((int(4*N/C1)-int(3*N/C1))/2),:,dim].T, 'b--', label="i="+str((int(3*N/C1)))+","+str((int(3*N/C1))+round((int(4*N/C1)-int(3*N/C1))/2)))
    plt.vlines(69, -3,3, linestyles='dashed', colors='r')
    plt.title("Group 1 3 & Group 1 4, merge")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    return

plot_fun(0)
# plt.plot(Actions[0])
# %% environment setup
# create folder under seed if not existing
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
def setpath(trans_setting, K=None, init=None):
    os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
    # if K is not None:
    #     path_name = 'results/2x3/K'+str(K)+'/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) +\
    #             '_kappa' + str(kappa) + '_N' + str(N) + '_T'+ str(T)+'clusterws'+str(clustering_warm_start) + 'signal' + str(signal)
    # else:
    path_name = 'results/2x3/'+init+'/K'+str(K)+'/coef0'+   re.sub("\\,", "", re.sub("\\ ", "", re.sub("\\.", "", re.sub("\\]","", re.sub("\\[", "", str(coef[0]))))))+\
    '/coef1'+   re.sub("\\ ", "",re.sub("\\.", "", re.sub("\\]","", re.sub("\\[", "", re.sub("\\, ", "", str(coef[1]))))))+\
        '/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) +\
                '_N' + str(N) + '_T'+ str(T)+'clusterws'+str(clustering_warm_start) + 'signal' + str(signal) + 'cov' + str(cov)
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    # path_name += '/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) + \
    #              '_kappa' + str(kappa) + '_N' + str(N) + '_1d_' + str(seed)
    # if not os.path.exists(path_name):
    #     os.makedirs(path_name, exist_ok=True)
    os.chdir(path_name)

def save_data(out, seed, K_list=None, init=None):
    if type(K_list) is range or type(K_list) is list:
        for k in K_list:
            setpath(trans_setting,k,init)
            tmp = out.models[K_list.index(k)]
            file_name = "seed_"+str(seed)+".dat"
            with open(file_name, "wb") as f:
                pickle.dump({'iter_num' : tmp.iter_num, 'group':tmp.g_index,
                             'changepoint' :tmp.changepoints, 
                             'changepoint_eachiter':tmp.changepoint_eachiter,
                             'g_index_eachiter':tmp.g_index_eachiter,
                             'loss':tmp.loss,
                             "ic":tmp.IC,
                             "K":k}, f)
    else:
        setpath(trans_setting,K_list, init)
        tmp = out
        file_name = "seed_"+str(seed)+".dat"
        with open(file_name, "wb") as f:
            pickle.dump({'iter_num' : tmp.iter_num, 'group':tmp.g_index,
                         'changepoint' :tmp.changepoints, 
                         'changepoint_eachiter':tmp.changepoint_eachiter,
                         'g_index_eachiter':tmp.g_index_eachiter,
                         'loss':tmp.loss,
                         "ic":tmp.IC,
                         "K":K_list}, f)
            
# direct the screen output to a file
# stdoutOrigin = sys.stdout
# sys.stdout = open("log.txt", "w")
# print("\nName of Python script:", sys.argv[0])
# sys.stdout.flush()

# sys.stdout = open("log_cluster.txt", "w")
# print("\nName of Python script:", sys.argv[0])
# sys.stdout.flush()
# %% evaluation function
def evaluate(changepoints_true, g_index, predict, N, T):
    '''
    g_index : predicted group index
    predict : predicted changepoints
    '''
    changepoints_true = changepoints_true.reshape(predict.shape)
    changepoint_err = np.mean(np.abs(predict - changepoints_true)/T)
    cluster_err = adjusted_rand_score(changepoints_true.flatten(), g_index.flatten())
    return changepoint_err, cluster_err
def evaluate_Klist(K_list, out,changepoints_true, N, T):
    changepoint_err_list=[None] * len(K_list)
    cluster_err_list=[None] * len(K_list)
    for k in K_list:
        tmp = out.models[K_list.index(k)]
        changepoint_err_list[K_list.index(k)], cluster_err_list[K_list.index(k)] = evaluate(changepoints_true, tmp.g_index, tmp.changepoints, N, T)
    return np.array(changepoint_err_list), np.array(cluster_err_list)

#%% simu: test ic multi-robust
N=36
M=50
C=10
T=100
K_list = range(2, 6)
# K_list = None
C3=3/2
p=2
loss_mat = np.zeros([M, len(K_list)])
Kl_mat = np.zeros([M, len(K_list)])
c_mat = np.zeros([M, len(K_list)])
ic_mat = np.zeros([M, len(K_list)])
example = "cdist"
bestK_list = [None]*M
g_index_true = np.append(np.zeros(int(N/3)), np.ones(int(N/3)))
g_index_true = np.append(g_index_true , 2*np.ones(int(N/3)))
clustering_warm_start=1

  #%% simu1: init changepoint separatly + K=3
init = "changepoint_separ"
changepoint_err_list_cpsep = []
cluster_err_list_cpsep = []
C3=3/2
for seed in range(10,50):
    print('==== seed',seed,'=====')
    States, Rewards, Actions, changepoints_true, g_index_true = gen_dat(N, T, K, 
                                                          coef, signal,None,
                                                          trans_setting,seed + 100)
    # startTime = datetime.now()
    # out = mean_detect.fit_tuneK(range(1,6),States, Actions, example="cdist", seed = seed, nthread=nthread, max_iter=5)
    out = mean_detect.fit(States, Actions, K=3, C3=C3, 
                          clustering_warm_start=clustering_warm_start,
                          example="cdist", seed = seed, nthread=nthread, max_iter=5)
    save_data(out, seed, K_list=3, init=init)
    print(out.changepoints)
    # runtimeone = datetime.now() - startTime
    # out = mean_detect.fit_tuneK(range(1,6),States, Actions, example="cdist", seed = seed, nthread=nthread, max_iter=5)
    # runtime.append(runtimeone)
    # print(out.K)
    changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), out[1].squeeze(),
                                            out[2].squeeze(), N, T)
    changepoint_err_list_cpsep.append(changepoint_err)
    cluster_err_list_cpsep.append(cluster_err)
    # iter_num.append(out[0])
    # print(out[2])
    print(changepoint_err)
    print(cluster_err)
print(sum(changepoint_err_list_cpsep)/len(changepoint_err_list_cpsep))
print(sum(cluster_err_list_cpsep)/len(cluster_err_list_cpsep))
  #%% simu2: init changepoint 
init_list = ["changepoint_no", "changepoints_random"]
changepoint_err_cpmis = [[] for i in range(len(init_list))]
cluster_err_cpmis = [[] for i in range(len(init_list))]

for init in init_list:
    # init = "nochangepoint"
    print("init",init)
    for seed in range(10,50):
        print("======= seed",seed, "==========")
        States, Rewards, Actions, changepoints_true,g_index_true = gen_dat(N=N, T=T, K=3, coef=coef, 
                                                              signal =signal,changepoint_list=None,
                                                              trans_setting=trans_setting,seed=seed + 100)
        if init == "changepoint_no":
            changepoints_init = np.zeros([N,1]) 
        elif init == "changepoints_random":
            changepoints_init = np.random.choice(range(T-1), size = N)
        elif init == "separ":
            changepoints_init = None
        out = mean_detect.fit(States, Actions, K=3, example="cdist", 
                              changepoints_init=changepoints_init, clustering_warm_start=clustering_warm_start,
                              seed = seed, C3=C3, nthread=nthread, max_iter=5)
        save_data(out, seed, K_list=3, init=init)
        print(out.changepoints)
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), out[1].squeeze(),
                                                out[2].squeeze(), N, T)
        changepoint_err_cpmis[init_list.index(init)].append(changepoint_err)
        cluster_err_cpmis[init_list.index(init)].append(cluster_err)
        # iter_num.append(out[0])
        # print(out[2])
        print(changepoint_err)
        print(cluster_err)
    print(np.mean(changepoint_err_cpmis[init_list.index(init)]))
    print(np.mean(cluster_err_cpmis[init_list.index(init)]))

for i in range(len(init_list)):
    print("====",init_list[i],"====")
    print(np.mean(changepoint_err_cpmis[i]))
    print(np.mean(cluster_err_cpmis[i]))
  #%% simu3: init consistent clustering with K = 3 
init = "clustering_Kmeans"
changepoint_err_km = []
cluster_err_km = []
C3=3/2
for seed in range(10,50):
    print("======= seed",seed, "==========")
    States, Rewards, Actions, changepoints_true,g_index_true = gen_dat(N=N, T=T, K=3, coef=coef, 
                                                          signal =signal,changepoint_list=None,
                                                          trans_setting=trans_setting,seed=seed + 100)
    np.random.seed(seed+500)
    out = mean_detect.fit(States, Actions, K=3, init_cluster_range=10,
                          example="cdist",init="clustering", seed = seed,
                          C3 = C3,clustering_warm_start=clustering_warm_start,
                          nthread=nthread, max_iter=5)
    # print(out.changepoints)
    save_data(out, seed, K_list=3,init=init)
    changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), out[1].squeeze(),
                                            out[2].squeeze(), N, T)
    changepoint_err_km.append(changepoint_err)
    cluster_err_km.append(cluster_err)
    print(changepoint_err)
    print(cluster_err)
   
print(np.mean(changepoint_err_km))
print(np.mean(cluster_err_km))
# print(np.mean(bestK_list[1:5]))
  #%% simu4: init random clustering with K_list 
init = "randomclustering"
K_list = range(2,6)
K_list = [5]
changepoint_err_mat = np.zeros([len(K_list), M])
cluster_err_mat = np.zeros([len(K_list), M])
C3=3/2 #1+2/3
# C3=9/8
for seed in range(10):
    print("======= seed",seed, "==========")
    States, Rewards, Actions, changepoints_true,g_index_true = gen_dat(N=N, T=T, K=3, coef=coef, 
                                                          signal =signal,changepoint_list=None,
                                                          trans_setting=trans_setting,seed=seed + 100)
    np.random.seed(seed+500)
    g_index_init_list = [np.random.choice(range(K), size = N) for K in K_list]
    # g_index_init_list[K_list.index(3)] = None
    out = mean_detect.fit_tuneK(K_list,States, Actions, init="clustering", example="cdist", 
                                seed = seed, nthread=nthread, max_iter=5,
                                C3=C3,init_cluster_range=10,
                                C=C,changepoints_init = None, 
                                g_index_init_list=g_index_init_list,
                                clustering_warm_start=clustering_warm_start)
    # bestK_list[seed]= out.K
    # print("best K",out.K)
    # print("changepoints3",out.models[0].changepoints)
    # print("group", out.models[0].g_index)
    # print("changepoints3", np.hstack([out.models[0].changepoints, 
    #                                   out.models[1].changepoints,
    #                                   out.models[2].changepoints]))
    # print("group", np.vstack([out.models[0].g_index, 
    #                                   out.models[1].g_index,
    #                                   out.models[2].g_index]))

    # print('cplist', out.models[1].changepoint_eachiter)
    # print('glist',out.models[1].g_index_eachiter)
    # print("changepoints", out3.changepoints)
    # plt.figure()
    # plt.plot(out.models[1].g_index)
    # plt.show()
    # plt.figure()
    # plt.plot(out.models[1].changepoints)
    # plt.show()
    # save_data(out, seed, K_list=K_list,init=init)
    changepoint_err_mat[:,seed],cluster_err_mat[:,seed]=evaluate_Klist(K_list, out,changepoints_true, N, T)
    print('cp',changepoint_err_mat[:,seed])
    print('adi',cluster_err_mat[:,seed])
    # print('loss',out.loss_model)
    
    
print('==using clustering warm start==')
print('K', list(K_list))
print('change point err',np.mean(changepoint_err_mat[:,:10], axis=1))
print('ARI',np.mean(cluster_err_mat[:,:10], axis=1))
# print('==no clustering warm start==')
# print('K', list(K_list))
# print('change point err',np.mean(changepoint_err_mat1[:,:10], axis=1))
# print('ARI',np.mean(cluster_err_mat1[:,:10], axis=1))
# print(np.mean(bestK_list[1:5]))
  #%% simu5: Oracle: changepoint_true / g_index_true init
init_list = ["changepoints","clustering"]
K_list = None
changepoint_err_list_ora = [[0]*M for i in range(2)]
cluster_err_list_ora = [[0]*M for i in range(2)]

for init in init_list:
    print('init',init)
    for seed in range(30,34):
        print("======= seed",seed, "==========")
        States, Rewards, Actions, changepoints_true, g_index_true = gen_dat(N=N, T=T, K=3, coef=coef, 
                                                              signal =signal,changepoint_list=None,
                                                              trans_setting=trans_setting,seed=seed + 100)
        if init == "changepoints":
            g_index_init = None
            changepoints_init = changepoints_true
            # sys.stdout = open("log_cp1.txt", "w")
            # print("\nName of Python script:", sys.argv[0])
            # sys.stdout.flush()
        elif init == "clustering":
            g_index_init = g_index_true
            changepoints_init = None

        out3 = mean_detect.fit(States, Actions, K=3, example="cdist", seed = seed, 
                                nthread=nthread, max_iter=5, init = init,
                                changepoints_init = changepoints_init,C3 = C3,
                                clustering_warm_start=clustering_warm_start,
                                init_cluster_range=10,
                                g_index_init = g_index_init)
        print(out3.changepoints)
        # print(out3.changepoint_eachiter)
        print(out3.g_index)
        # save_data(out3, seed, K_list=3,init=init+"_ora")
        changepoint_err_list_ora[init_list.index(init)][seed], cluster_err_list_ora[init_list.index(init)][seed] = evaluate(changepoints_true.squeeze(), out3[1].squeeze(),
                                                out3[2].squeeze(), N, T)  
        print('cp err', changepoint_err_list_ora[init_list.index(init)][seed])
        print('adi',cluster_err_list_ora[init_list.index(init)][seed])
        
      
# print(np.mean(changepoint_err_mat, axis=1))
# print(np.mean(cluster_err_mat, axis=1))
# print(np.mean(bestK_list))
for i in range(len(init_list)):
    print("== init", init_list[i], "==")
    print('changepoint_err', round(np.mean(changepoint_err_list_ora[i]),3),'ARI', round(np.mean(cluster_err_list_ora[i]),3))
    # print('ARI', np.mean(cluster_err_list_ora1[i][:10]))
# print("===== no clustering warm start =====")
# for i in range(len(init_list)):
#     print("== init", init_list[i], "==")
#     print('changepoint_err', round(np.mean(changepoint_err_list_ora[i][:10]),3),'ARI',round(np.mean(cluster_err_list_ora_w0[i][:10]),3))
    # print('ARI',np.mean(cluster_err_list_ora[i][:10]))
#%% analyse results
include_path = 0

#%% analyse results 1 -- init changepoint separately K = 3/ changepoint_no/ changepoints random
C_list =[0, 0.1, 1, 10,50,100,200,500, 1000]
init_list = ["changepoint_separ", "changepoint_no","changepoints_random"]
K_list = [3]
ic_list = np.zeros(len(init_list))
loss_mat_cp = np.zeros([M, len(init_list)])
Kl_mat = np.zeros([M, len(init_list)])
c_mat = np.zeros([M, len(init_list)])
changepoint_err_mat_cp = np.zeros([M,len(init_list)])
cluster_err_mat_cp = np.zeros([M,len(init_list)])
iter_mat_cp = np.zeros([M,len(init_list)])

for init in init_list:
    for seed in range(M):
        for K in K_list:
            setpath(trans_setting, K = K, init=init)
            file_name = "seed_"+str(seed)+".dat"
            pkl_file = open(file_name, 'rb')
            t = pickle.load(pkl_file)
            pkl_file.close()
            # t['loss']
            # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
            #                               t['group'], N, T, K, C)
            loss_mat_cp[seed, init_list.index(init)] = t['loss']
            Kl_mat[seed, init_list.index(init)] =  K*np.log(np.sum(T-1 -t['changepoint']))
            Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
            c_mat[seed, init_list.index(init)] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
            changepoint_err_mat_cp[seed,init_list.index(init)],cluster_err_mat_cp[seed,init_list.index(init)] = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                    t['changepoint'].squeeze(), N, T)
            iter_mat_cp[seed,init_list.index(init)] = t['iter_num']
           


res_diffC_cp = [None] * len(C_list)
for C in C_list:
    # bestK_list = np.zeros(M)
    changepoint_err_list = []
    cluster_err_list= []
    iter_num =[]
    # loss_list = []
    for seed in range(M):
        ic_list = loss_mat_cp[seed, :] - Kl_mat[seed, :] + c_mat[seed,:]*C
        setpath(trans_setting, K = 3, init=init_list[np.argmax(ic_list)])
        file_name = "seed_"+str(seed)+".dat"
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                t['changepoint'].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        # loss_list.append(loss_mat_cp[seed, np.argmax(ic_list)])
        iter_num.append(t['iter_num'])
        pkl_file.close()
    res_diffC_cp[C_list.index(C)] = {"C":C,
                    "changepoint_err": np.mean(changepoint_err_list),
                    "ARI": np.mean(cluster_err_list),
                    # "loss":np.mean(loss_list),
                    "iter_num":np.mean(iter_num),
                    # "bestK":np.mean(bestK_list),
                    "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                    'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                    'iter_num_var':np.std(iter_num)/np.sqrt(M)
                    # 'bestK_var':np.std(bestK_list)/np.sqrt(M)
                    }

dat_cp = pd.DataFrame(res_diffC_cp)
dat_cp = round(dat_cp, 3)
for i in range(1,4):
    dat_cp.iloc[:,i] = [str(a) +'('+ str(b)+")" for a,b in zip(dat_cp.iloc[:,i],dat_cp.iloc[:,i+3])]
col = dat_cp.columns.tolist()
dat_cp = dat_cp[[col[0], col[4]]+ col[1:4]]
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
# file_name="results/2x3/tables/res0603.xlsx"
# dat_cp.to_excel(file_name, index=False,sheet_name = "cp_ic")  

# all performances
dat_all_cp = pd.DataFrame({"init":init_list,
              "changepoint_err":np.mean(changepoint_err_mat_cp, axis=0),
              "ARI":np.mean(cluster_err_mat_cp, axis=0),
              "iter_num":np.mean(iter_mat_cp, axis=0),
              "cp_var":np.std(changepoint_err_mat_cp, axis=0)/np.sqrt(M),
              "cluster_var":np.std(cluster_err_mat_cp, axis=0)/np.sqrt(M),
              "iter_num_var":np.std(iter_mat_cp,axis=0)/np.sqrt(M)})
dat_all_cp.iloc[:,1:7] = round(dat_all_cp.iloc[:,1:7], 3)
for i in range(1,4):
    dat_all_cp.iloc[:,i] = [str(a) +'('+ str(b)+")" for a,b in zip(dat_all_cp.iloc[:,i],dat_all_cp.iloc[:,i+3])]
col = dat_all_cp.columns.tolist()
dat_all_cp = dat_all_cp[col[:4]]
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
file_name="results/2x3/tables/res0603_2.xlsx"
# with pd.ExcelWriter(file_name) as writer:
#     dat_cp.to_excel(writer, sheet_name="IC", index=False)
#     dat_all_cp.to_excel(writer, sheet_name="all", index=False)
# with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#     dat_all_cp.to_excel(writer, sheet_name="cp all", index=False)

#%% analyse results 2 -- oracle
K = 3
oracle_list = ["changepoints_ora", "clustering_ora"]
res_oracle = [None] * len(oracle_list)
for init in oracle_list :
    ic_list = np.zeros(1)
    loss_mat = np.zeros([M, 1])
    Kl_mat = np.zeros([M, 1])
    c_mat = np.zeros([M, 1])
                
    bestK_list = np.zeros(M)
    changepoint_err_list = []
    cluster_err_list= []
    iter_num =[]
    for seed in range(30,34):
        setpath(trans_setting, K = K, init=init)
        file_name = "seed_"+str(seed)+".dat"
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        # print('seed',seed,t['iter_num'])
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                t['changepoint'].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        iter_num.append(t['iter_num'])
        pkl_file.close()
    res_oracle[oracle_list.index(init)] = {"oracle":init,
                                  "changepoint_err": np.mean(changepoint_err_list),
                                  "ARI": np.mean(cluster_err_list),
                                  "iter_num":np.mean(iter_num),
                                  "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                                  'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                                  'iter_num_var':np.std(iter_num)/np.sqrt(M)}


dat_oracle = pd.DataFrame(res_oracle)
dat_oracle = round(dat_oracle, 3)
for i in range(1,4):
    dat_oracle.iloc[:,i] = [str(a) +'('+ str(b)+")" for a,b in zip(dat_oracle.iloc[:,i],dat_oracle.iloc[:,i+3])]
col = dat_oracle.columns.tolist()
dat_oracle = dat_oracle[[col[0]]+ col[1:4]]
# os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
# #     dat_all_cp.to_excel(writer, sheet_name="all", index=False)
# with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#     dat_oracle.to_excel(writer, sheet_name="oracle", index=False)
# # dat_oracle.to_excel(file_name)  
#%% analyse result 3 -- random cluster K_list all K's performance + consistent Kmeans
K_list = range(2, 6)
init_list= ["randomclustering", "clustering_Kmeans"]
## ic 
loss_mat_K = np.zeros([M, len(K_list)+1])
Kl_mat = np.zeros([M, len(K_list)+1])
c_mat = np.zeros([M, len(K_list)+1])
changepoint_err_mat_K = np.zeros([M,len(K_list)+1])
cluster_err_mat_K = np.zeros([M,len(K_list)+1])
iter_mat_K = np.zeros([M,len(K_list)+1])

for init in init_list:
    if init == "randomclustering":
        # for K in K_list:
        for seed in range(M):
            # print('===== seed',seed,'======')
            for K in K_list:
                setpath(trans_setting, K = K, init=init)
                file_name = "seed_"+str(seed)+".dat"
                pkl_file = open(file_name, 'rb')
                t = pickle.load(pkl_file)
                pkl_file.close()
                # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
                #                               t['group'], N, T, K, C)
                print('===== K',K,'cp',t['changepoint'], 'g', t['group'])
                print('loss', t['loss'])
                loss_mat_K[seed, K_list.index(K)] = t['loss']
                changepoint_err_mat_K[seed,K_list.index(K)],cluster_err_mat_K[seed,K_list.index(K)] = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                     t['changepoint'].squeeze(), N, T)
                iter_mat_K[seed,K_list.index(K)] = t['iter_num']
               
                Kl_mat[seed, K_list.index(K)], c_mat[seed, K_list.index(K)]=uti.paramInIC(t, N, K, T)
                print('cperr',changepoint_err_mat_K[seed,K_list.index(K)],'ari',cluster_err_mat_K[seed,K_list.index(K)],'Kl',Kl_mat[seed, K_list.index(K)], 'c',c_mat[seed, K_list.index(K)])
    elif init == "clustering_Kmeans":
        for seed in range(M):
            setpath(trans_setting, K = 3, init=init)
            file_name = "seed_"+str(seed)+".dat"
            pkl_file = open(file_name, 'rb')
            t = pickle.load(pkl_file)
            # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
            #                               t['group'], N, T, K, C)
            loss_mat_K[seed, len(K_list)] = t['loss']
            Kl_mat[seed, len(K_list)] =  K*np.log(np.sum(T-1 -t['changepoint']))
            Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
            c_mat[seed, len(K_list)] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
            changepoint_err_mat_K[seed,len(K_list)],cluster_err_mat_K[seed,len(K_list)] = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                    t['changepoint'].squeeze(), N, T)
            iter_mat_K[seed,len(K_list)] = t['iter_num']
            pkl_file.close()

res_diffC_K = [None] * len(C_list)
for C in C_list:
    bestK_list = np.zeros(M)
    changepoint_err_list = []
    cluster_err_list= []
    iter_num =[]
    # loss_list = []
    for seed in range(M):
        ic_list = loss_mat_K[seed,:] - Kl_mat[seed,:] + c_mat[seed,:]*C
        if np.argmax(ic_list) == len(K_list):
            bestK = 3
            init = "clustering_Kmeans"
        else:
            bestK = K_list[np.argmax(ic_list)]
            init = "randomclustering"
        bestK_list[seed] = bestK
        setpath(trans_setting, K = bestK, init=init)
        file_name = "seed_"+str(seed)+".dat"
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                t['changepoint'].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        # loss_list.append(loss_mat_cp[seed, np.argmax(ic_list)])
        if t['iter_num'] > 3:
            print('seed',seed,'iter_num', t['iter_num'])
        iter_num.append(t['iter_num'])
        pkl_file.close()
    res_diffC_K[C_list.index(C)] = {"C":C,
                                     "bestK":np.mean(bestK_list),
                    "changepoint_err": np.mean(changepoint_err_list),
                    "ARI": np.mean(cluster_err_list),
                    "iter_num":np.mean(iter_num),
                    # "bestK":np.mean(bestK_list),
                    'bestK_var':np.std(bestK_list)/np.sqrt(M),
                    "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                    'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                    'iter_num_var':np.std(iter_num)/np.sqrt(M)
                    }

dat_K = pd.DataFrame(res_diffC_K)
dat_K = round(dat_K, 3)
for i in range(1,5):
    dat_K.iloc[:,i] = [str(a) +'('+ str(b)+")" for a,b in zip(dat_K.iloc[:,i],dat_K.iloc[:,i+4])]
col = dat_K.columns.tolist()
dat_K = dat_K[col[:5]]

## all performance
dat_all_K = pd.DataFrame({"K":list(K_list) + [3],
                      "changepoint_err":np.mean(changepoint_err_mat_K, axis=0),
                      "ARI":np.mean(cluster_err_mat_K, axis=0),
                      "iter_num":np.mean(iter_mat_K, axis=0),
                      "cp_var":np.std(changepoint_err_mat_K, axis=0)/np.sqrt(M),
                      "cluter_var":np.std(cluster_err_mat_K, axis=0)/np.sqrt(M),
                      "iter_num_var":np.std(iter_mat_K,axis=0)/np.sqrt(M)})
# dat_K = pd.DataFrame(dat_K, columns=['K','changepoint_err', 'cluster_err',
#                                      'loss','iter_num','cp_var','cluster_var',
#                                      'loss_var','iter_num_var'])
dat_all_K = round(dat_all_K, 3)
for i in range(1,4):
    dat_all_K.iloc[:,i] = [str(a)+"("+str(b)+")" for a, b in zip(dat_all_K.iloc[:,i], dat_all_K.iloc[:,i+3])]

cols = dat_all_K.columns.tolist()
dat_all_K = dat_all_K[cols[:4]]
dat_all_K = dat_all_K.iloc[[4]+list(range(len(K_list)))]
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")

#%% analyse result 4 -- ic best K when the 2 kinds of initial estimators are combined
# C_list =[0,5,8,10,20, 50]
## ic
init_list = ["changepoints_ora", "clustering_ora","changepoint_separ", "changepoint_no", "changepoints_random", 'randomclustering']
# init_list = ["changepoints_ora", "clustering_ora",'randomclustering']
C_list = np.arange(0,10,1).tolist()
K_list = range(2, 6)
ic_list = [None] * len(init_list)
loss_mat = [None] * len(init_list)
Kl_mat = [None] * len(init_list)
c_mat = [None] * len(init_list)
for init in init_list:
    if init == "randomclustering":
        loss_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
        Kl_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
        c_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
        for seed in range(M):
            for K in K_list:
                setpath(trans_setting, K = K, init=init)
                file_name = "seed_"+str(seed)+".dat"
                pkl_file = open(file_name, 'rb')
                t = pickle.load(pkl_file)
                # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
                #                               t['group'], N, T, K, C)
                loss_mat[init_list.index(init)][seed, K_list.index(K)] = t['loss']# / np.mean(T - t['changepoint'] - 1)
                # Kl_mat[init_list.index(init)][seed, K_list.index(K)] =  K*np.log(np.sum(T-1 -t['changepoint']))
                # Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
                # c_mat[init_list.index(init)][seed, K_list.index(K)] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
                pkl_file.close()
                Kl_mat[init_list.index(init)][seed, K_list.index(K)], c_mat[init_list.index(init)][seed, K_list.index(K)]=uti.paramInIC(t, N, K, T)
    else:
        loss_mat[init_list.index(init)] = np.zeros([M, 1])
        Kl_mat[init_list.index(init)] = np.zeros([M, 1])
        c_mat[init_list.index(init)] = np.zeros([M, 1])
        for seed in range(M):
            setpath(trans_setting, K = 3, init=init)
            file_name = "seed_"+str(seed)+".dat"
            pkl_file = open(file_name, 'rb')
            t = pickle.load(pkl_file)
            # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
            #                               t['group'], N, T, K, C)
            # loss = mean_detect.goodnessofClustering(States, N, T, 3, t['changepoint'], Actions, t['group'])
            loss_mat[init_list.index(init)][seed, 0] = t['loss'] #/ np.mean(T - t['changepoint'] - 1)
            # Kl_mat[init_list.index(init)][seed, 0] =  K*np.log(np.sum(T-1 -t['changepoint']))
            # Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
            # c_mat[init_list.index(init)][seed, 0] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
            pkl_file.close()
            Kl_mat[init_list.index(init)][seed, 0],c_mat[init_list.index(init)][seed, 0] = uti.paramInIC(t, N, 3, T)



res_diffC_all = [None] * len(C_list)
for C in C_list:
    init_count = np.zeros(len(init_list))
    K_count = np.zeros(len(K_list))
    bestK_list = np.zeros(M)
    changepoint_err_list = []
    cluster_err_list= []
    iter_num =[]
    for seed in range(M):
        # print('seed', seed)
        for init in init_list:
            # print('====init:', init,"====")
            if init == "randomclustering":
                # print('loss_mat[',init_list.index(init),'][seed, :]',loss_mat[init_list.index(init)][seed, :], 'Kl', Kl_mat[init_list.index(init)][seed, :] ,'cmat', c_mat[init_list.index(init)][seed,:]*C)
                # for C in np.arange(10.433, 10.44, 0.00001):
                    # print('====C',C,"=====")
                    # print(loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C)
                    # print('**maxK', K_list[np.argmax(loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C)])
                # loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C
                ic_list[init_list.index(init)] = loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C
                clusterK = K_list[np.argmax(ic_list[init_list.index('randomclustering')])]
                # print('ic_list[init_list.index("randomclustering")]',ic_list[init_list.index('randomclustering')])
                ic_list[init_list.index('randomclustering')] = np.max(ic_list[init_list.index('randomclustering')])
            else:    
                # print('loss_mat[',init_list.index(init),'][seed, :]',loss_mat[init_list.index(init)][seed, :], 'Kl', Kl_mat[init_list.index(init)][seed, :] ,'cmat', c_mat[init_list.index(init)][seed,:]*C)
                ic_list[init_list.index(init)] = (loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C)[0]
        # print('ic_list' ,ic_list)
        init = init_list[np.where(ic_list == np.max(ic_list))[0][0]]
        init_count[np.where(ic_list == np.max(ic_list))[0][0]] = init_count[np.where(ic_list == np.max(ic_list))[0][0]] + 1
        if init == "randomclustering":
            bestK = clusterK
        else:
            bestK = 3
        K_count[K_list.index(bestK)] = K_count[K_list.index(bestK)] +1
        # print('**** init', init, 'K', bestK)
        # bestK = K_list[np.where(ic_list == np.max(ic_list))[1][0]]
        bestK_list[seed]=bestK
        setpath(trans_setting, K = bestK, init=init)
        file_name = "seed_"+str(seed)+".dat"
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        print(t['changepoint'])
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                t['changepoint'].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        # print('cp_err_best', changepoint_err, "ARI", cluster_err)
        cluster_err_list.append(cluster_err)
        iter_num.append(t['iter_num'])
        pkl_file.close()
        
    res_diffC_all[C_list.index(C)] = {"C":C,
                                  "changepoint_err": np.mean(changepoint_err_list),
                                  "ARI": np.mean(cluster_err_list),
                                  "iter_num":np.mean(iter_num),
                                  "bestK":np.mean(bestK_list),
                                  "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                                  'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                                  'iter_num_var':np.std(iter_num)/np.sqrt(M),
                                  'bestK_var':np.std(bestK_list)/np.sqrt(M),
                                  "init_count":init_count,
                                  "K_count":K_count}

dat_all = pd.DataFrame(res_diffC_all)
dat_all = round(dat_all, 3)
for i in range(1,5):
    dat_all.iloc[:,i] = [str(a) +'('+ str(b)+")" 
                         for a,b in zip(dat_all.iloc[:,i],dat_all.iloc[:,i+4])]
col = dat_all.columns.tolist()
dat_all = dat_all[[col[0], col[4]]+ col[1:4]]
print(dat_all)
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
# file_name="results/2x3/res_combined_2_init_estimators"+".xlsx"
# dat_all.to_excel(file_name)  
#%% analyse result 5 -- ic best K no oracle
# C_list =[0, 0.1, 1, 10,50,100,200,300,400,500, 1000]
## ic
init_list = ["changepoint_separ", "changepoint_no", "changepoints_random", 'randomclustering']

K_list = range(2, 6)
ic_list = [None] * len(init_list)
loss_mat = [None] * len(init_list)
Kl_mat = [None] * len(init_list)
c_mat = [None] * len(init_list)
for init in init_list:
    if init == "randomclustering":
        loss_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
        Kl_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
        c_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
        for seed in range(M):
            for K in K_list:
                setpath(trans_setting, K = K, init=init)
                file_name = "seed_"+str(seed)+".dat"
                pkl_file = open(file_name, 'rb')
                t = pickle.load(pkl_file)
                # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
                #                               t['group'], N, T, K, C)
                loss_mat[init_list.index(init)][seed, K_list.index(K)] = t['loss']# / np.mean(T - t['changepoint'] - 1)
                # Kl_mat[init_list.index(init)][seed, K_list.index(K)] =  K*np.log(np.sum(T-1 -t['changepoint']))
                # Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
                # c_mat[init_list.index(init)][seed, K_list.index(K)] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
                pkl_file.close()
                Kl_mat[init_list.index(init)][seed, K_list.index(K)], c_mat[init_list.index(init)][seed, K_list.index(K)]=uti.paramInIC(t, N, K, T)
    else:
        loss_mat[init_list.index(init)] = np.zeros([M, 1])
        Kl_mat[init_list.index(init)] = np.zeros([M, 1])
        c_mat[init_list.index(init)] = np.zeros([M, 1])
        for seed in range(M):
            setpath(trans_setting, K = 3, init=init)
            file_name = "seed_"+str(seed)+".dat"
            pkl_file = open(file_name, 'rb')
            t = pickle.load(pkl_file)
            # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
            #                               t['group'], N, T, K, C)
            # loss = mean_detect.goodnessofClustering(States, N, T, 3, t['changepoint'], Actions, t['group'])
            loss_mat[init_list.index(init)][seed, 0] = t['loss'] #/ np.mean(T - t['changepoint'] - 1)
            # Kl_mat[init_list.index(init)][seed, 0] =  K*np.log(np.sum(T-1 -t['changepoint']))
            # Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
            # c_mat[init_list.index(init)][seed, 0] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
            pkl_file.close()
            Kl_mat[init_list.index(init)][seed, 0],c_mat[init_list.index(init)][seed, 0] = uti.paramInIC(t, N, 3, T)


# C_list = [0,10,100,200,400]
res_diffC_noora = [None] * len(C_list)
for C in C_list:
    bestK_list = np.zeros(M)
    changepoint_err_list = []
    cluster_err_list= []
    iter_num =[]
    init_count = np.zeros(len(init_list))
    K_count = np.zeros(len(K_list))
    for seed in range(M):
        for init in init_list:
            if init == "randomclustering":
                ic_list[init_list.index(init)] = loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C
                clusterK = K_list[np.argmax(ic_list[init_list.index('randomclustering')])]
                ic_list[init_list.index('randomclustering')] = np.max(ic_list[init_list.index('randomclustering')])
            else:    
                ic_list[init_list.index(init)] = (loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C)[0]
        init = init_list[np.where(ic_list == np.max(ic_list))[0][0]]
        if init == "randomclustering":
            bestK = clusterK
        else:
            bestK = 3
        # bestK = K_list[np.where(ic_list == np.max(ic_list))[1][0]]
        bestK_list[seed]=bestK
        print('**** init', init, 'K', bestK)
        init_count[np.where(ic_list == np.max(ic_list))[0][0]] = init_count[np.where(ic_list == np.max(ic_list))[0][0]] + 1
        K_count[K_list.index(bestK)] = K_count[K_list.index(bestK)] +1
        setpath(trans_setting, K = bestK, init=init)
        file_name = "seed_"+str(seed)+".dat"
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                t['changepoint'].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        iter_num.append(t['iter_num'])
        pkl_file.close()
    res_diffC_noora[C_list.index(C)] = {"C":C,
                                  "changepoint_err": np.mean(changepoint_err_list),
                                  "ARI": np.mean(cluster_err_list),
                                  "iter_num":np.mean(iter_num),
                                  "bestK":np.mean(bestK_list),
                                  "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                                  'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                                  'iter_num_var':np.std(iter_num)/np.sqrt(M),
                                  'bestK_var':np.std(bestK_list)/np.sqrt(M)}

dat_all_noora = pd.DataFrame(res_diffC_noora)
dat_all_noora = round(dat_all_noora, 3)
for i in range(1,5):
    dat_all_noora.iloc[:,i] = [str(a) +'('+ str(b)+")" 
                         for a,b in zip(dat_all_noora.iloc[:,i],dat_all_noora.iloc[:,i+4])]
col = dat_all_noora.columns.tolist()
dat_all_noora = dat_all_noora[[col[0], col[4]]+ col[1:4]]
print(dat_all_noora)
# file_name="results/2x3/res_combined_2_init_estimators"+".xlsx"
# dat_all.to_excel(file_name)  

#%% write to excel
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
#%% write to excel
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
file_name="results/2x3/tables/res0609.xlsx"
with pd.ExcelWriter(file_name) as writer:
    dat_cp.to_excel(writer, sheet_name="cp ic", index=False)
    dat_all_cp.to_excel(writer, sheet_name="cp all", index=False)
    dat_oracle.to_excel(writer, sheet_name = "oracle", index=False)
    dat_K.to_excel(writer, sheet_name = "K ic", index=False)
    dat_all_K.to_excel(writer, sheet_name = "K all", index=False)
    dat_all.to_excel(writer, sheet_name = "all ic", index=False)
    dat_all_noora.to_excel(writer, sheet_name = "no ora ic", index=False)