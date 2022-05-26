'''
Simulate stationary 2-dimensional time series data in 4 scenarios:
transition: homogeneous; reward:
transition: homogeneous; reward: smooth
transition: piece-wise constant ; reward: homogeneous
transition: smooth; reward: homogeneous
'''
#!/usr/bin/python
import platform, sys, os, pickle, re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu/results")
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") # 引用模块的地址
import simu.compute_test_statistics as stat
import simu.simulate_data_pd as sim
import simu.simu_mean_detect as mean_detect
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
N = int(100)

np.random.seed(seed)
# %% simulate data
# terminal timestamp
T = 100
kappa = T
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

# width of smooth transition function
w = 0.01
delta = 1/10

# heter changpoints and means
K = 2
signal_list = [5, -5]

# parallel
nthread=3

coef =[[[0.1, 0.5, 0.5],[0.1, 0.25, 0.75]], [[0.1,0.0,0.75],[0,0.25,-0.5], [0.5, 0.1, 0.25]] 

def gen_dat(N, T, K, coef, changepoint_list=None,trans_setting="pwsonst2",
            seed=1):
    if changepoint_list is None:
        changepoint_list = [int(T/2) + int(0.1 * T) - 1, int(T/2) - int(0.1 * T) - 1] 
    changepoints_true = np.zeros([N, 1])
    States = np.zeros([N, T, p])
    Rewards = np.zeros([N, T-1])
    Actions = np.zeros([N, T-1])
    for i in range(K):
        changepoint = changepoint_list[i]
        sim_dat = sim.simulate_data(int(N/K), T, p, changepoint, delta)
        if trans_setting == 'pwconst2' and reward_setting == 'homo':
            def mytransition_function(t):
                return sim_dat.transition_pwconstant2(t, mean, cov, coef[i])
            def myreward_function(t):
                return sim_dat.reward_homo()
            States0, Rewards0, Actions0 = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed + changepoint_list.index(changepoint))
        elif trans_setting == 'smooth' and reward_setting == 'homo':
            def mytransition_function(t):
                return sim_dat.transition_smooth2(t, mean, cov, w, coef=coef[i])
            def myreward_function(t):
                return sim_dat.reward_homo()
            States0, Rewards0, Actions0 = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed + changepoint_list.index(changepoint))
        States[changepoint_list.index(changepoint)*int(N/K)+ 0:changepoint_list.index(changepoint)*int(N/K)+int(N/K), :, :] = States0
        Rewards[changepoint_list.index(changepoint)*int(N/K)+ 0:changepoint_list.index(changepoint)*int(N/K)+int(N/K), :] = Rewards0
        Actions[changepoint_list.index(changepoint)*int(N/K)+ 0:changepoint_list.index(changepoint)*int(N/K)+int(N/K), :] = Actions0
        changepoints_true[changepoint_list.index(changepoint)*int(N/2)+ 0:changepoint_list.index(changepoint)*int(N/2)+int(N/2), ] = changepoint_list[i] * np.ones([int(N/K),1])
    # normalize state variables
    # def transform(x):
    #     return (x - np.mean(x)) / np.std(x)
    # for i in range(p):
    #     States[:,:,i] = transform(States[:,:,i])
    return States, Rewards, Actions, changepoints_true


#%% plot trajectory
fig, ax = plt.subplots()
# plt.figure(figsize=(10, 3))
# plt.ylim(-100, 100)
# plt.ylim(-5, 5)
plt.plot(X[:50], 'ro')
plt.plot(X[50:], 'bo')
plt.plot(States[[0,1,2],:,0].T)
# plt.vlines([changepoints_true[[0]],changepoints_true[[50]]], -1.4,0, linestyles='dashed', colors='blue')
ax.vlines([39,59], 0,1e8, linestyles='dashed', colors='blue')
plt.show()
# plt.plot(changepoints_true[[1]], 'b--')
plt.plot(cusum_forward[0],'r',label="cumsum forward")
plt.plot(cusum_backward[0],'b',label="cumsum backward")

plt.legend(bbox_to_anchor=(1.0, 1), borderaxespad=0.)

plt.plot(States[np.s_[::20],:,0].T)
plt.plot(Actions[0])
# %% environment setup
# create folder under seed if not existing
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
def setpath(trans_setting, example="mean", shift="mean"):
    os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
    path_name = 'results/'+example+'/'+shift+'/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) +\
                '_kappa' + str(kappa) + '_N' + str(N) + '_1d' 
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    # path_name += '/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) + \
    #              '_kappa' + str(kappa) + '_N' + str(N) + '_1d_' + str(seed)
    # if not os.path.exists(path_name):
    #     os.makedirs(path_name, exist_ok=True)
    os.chdir(path_name)

# direct the screen output to a file
# stdoutOrigin = sys.stdout
# sys.stdout = open("log.txt", "w")
# print("\nName of Python script:", sys.argv[0])
# sys.stdout.flush()
# %% evaluation function
def evaluate(changepoints_true, g_index, predict, N, T):
    '''
    g_index : predicted group index
    predict : predicted changepoints
    '''
    changepoint_err = np.mean(np.abs(predict - changepoints_true)/T)
    cluster_err = adjusted_rand_score(changepoints_true, g_index)
    return changepoint_err, cluster_err

# %% mean changepoint detection
M=100
shift = "linear"
result_mean = [{},{}]
for shift in ["abrupt","linear"]:
    print(shift)
    if shift=="abrupt":
        coef =[[0,0, 0, 0.5],[0,0,0,0.5]] # shift in mean
    elif shift=="linear":
        coef =[[0.5, 0.5, 0.5, -0.5],[0.5,1.5,0.5,0.5]] # shift linearly
    changepoint_err_list = []
    cluster_err_list = []
    iter_num = []
    runtime  =[]
    trans_setting = "pwconst2"
    
    setpath(trans_setting,example="mean", shift=shift)
    for seed in range(M):
        States, Rewards, Actions, changepoints_true = gen_dat(N, T, K, coef, signal_list, 
                                                              None,trans_setting,seed)
        startTime = datetime.now()
        out = mean_detect.fit(States, Actions, example="mean", seed = seed)
        runtimeone = datetime.now() - startTime
        runtime.append(runtimeone)
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), 
                                                out[1].squeeze(), out[2].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        iter_num.append(out[0])
        print(changepoint_err)
        file_name = "seed_"+str(seed)+".dat"
        with open(file_name, "wb") as f:
            pickle.dump({'iter_num' : out[0], 'result' :out[2],'runtime': runtimeone}, f)
    
    print(sum(changepoint_err_list)/len(changepoint_err_list))
    print(sum(cluster_err_list)/len(cluster_err_list))
    print(sum(iter_num)/len(iter_num)) 
    print(np.mean(runtime))
    tmp = {"changepoint_err":sum(changepoint_err_list)/len(changepoint_err_list),
                 "cluster_err":sum(cluster_err_list)/len(cluster_err_list),
                 "iter_num":sum(iter_num)/len(iter_num),
                 "run_time":np.mean(runtime)}
    if shift == "linear":
        result_mean[0] = tmp
    elif shift =="abrupt":
        result_mean[1]=tmp
    file_name = 'result.dat'
    with open(file_name, 'wb') as f:
        pickle.dump(tmp, f)
        
# out1 = mean_detect.fit(States, Actions, example="mean", seed = seed)
# out1.changepoints
# %% marginal distribution
M=100
shift = "linear"
example = "marginal"
result_marginal = [{},{}]
g_index = np.hstack([np.zeros(50), np.ones(50)])
for shift in ["abrupt","linear"]:
    print("shift", shift)
    if shift=="abrupt":
        coef =[[0,0, 0, 0.5],[0,0,0,0.5]] # shift in mean
    elif shift=="linear":
        coef =[[0.5, 0.5, 0.5, -0.5],[0.5,1.5,0.5,0.5]] # shift linearly
    changepoint_err_list = []
    cluster_err_list = []
    iter_num = []
    runtime = []
    trans_setting = "pwconst2"
    setpath(trans_setting,example="marginal_paperthreshold", shift=shift)
    for seed in range(M):
        States, Rewards, Actions, changepoints_true = gen_dat(N, T, K, coef, signal_list,None,
                                                              trans_setting,seed)
        startTime = datetime.now()
        out = mean_detect.fit(States, Actions, example=example, seed = seed, nthread=nthread)
        runtimeone = datetime.now() - startTime
        runtime.append(runtimeone)
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), out[1].squeeze(),
                                                out[2].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        iter_num.append(out[0])
        # print(out[2])
        print(changepoint_err)
        file_name = "seed_"+str(seed)+".dat"
        with open(file_name, "wb") as f:
            pickle.dump({'iter_num' : out[0], 'result' :out[2], 'runtime': runtimeone}, f)
            
    print(sum(changepoint_err_list)/len(changepoint_err_list))
    print(sum(cluster_err_list)/len(cluster_err_list))
    print(sum(iter_num)/len(iter_num)) 
    print(np.mean(runtime))
    tmp = {"changepoint_err":sum(changepoint_err_list)/len(changepoint_err_list),
                 "cluster_err":sum(cluster_err_list)/len(cluster_err_list),
                 "iter_num":sum(iter_num)/len(iter_num),
                 "run_time":np.mean(runtime)}
    print(tmp)
    if shift == "linear":
        result_marginal[0] = tmp
    elif shift =="abrupt":
        result_marginal[1]=tmp
    file_name = 'result.dat'
    with open(file_name, 'wb') as f:
        pickle.dump(tmp, f)
print(result_marginal)
# %% conditional distribution
M=100
shift = "linear"
result_condis = [{},{}]
for shift in ["linear", "abrupt"]:
    print("shift", shift)
    if shift=="abrupt":
        coef =[[0,0, 0, 0.5],[0,0,0,0.5]] # shift in mean
    elif shift=="linear":
        coef =[[0.5, 0.5, 0.5, -0.5],[0.5,1.5,0.5,0.5]] # shift linearly
    changepoint_err_list = []
    cluster_err_list = []
    iter_num = []
    runtime = []
    trans_setting = "pwconst2"
    setpath(trans_setting,example="cdist2", shift=shift)
    for seed in range(M):
        States, Rewards, Actions, changepoints_true = gen_dat(N, T, K, coef,None,
                                                              trans_setting,seed)
        startTime = datetime.now()
        out = mean_detect.fit(States, Actions, example="cdist", seed = seed, nthread=nthread, max_iter=5)
        runtimeone = datetime.now() - startTime
        out = mean_detect.fit_tuneK([1,2],States, Actions, example="cdist", seed = seed, nthread=nthread, max_iter=5)

        runtime.append(runtimeone)
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), out[1].squeeze(),
                                                out[2].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        iter_num.append(out[0])
        # print(out[2])
        print(changepoint_err)
        print(cluster_err)
        file_name = "seed_"+str(seed)+".dat"
        with open(file_name, "wb") as f:
            pickle.dump({'iter_num' : out[0], 'group':out[1],'changepoint' :out[2], 'runtime': runtimeone}, f)
            
    print(sum(changepoint_err_list)/len(changepoint_err_list))
    print(sum(cluster_err_list)/len(cluster_err_list))
    print(sum(iter_num)/len(iter_num)) 
    print(np.mean(runtime))
    tmp = {"changepoint_err":sum(changepoint_err_list)/len(changepoint_err_list),
                 "cluster_err":sum(cluster_err_list)/len(cluster_err_list),
                 "iter_num":sum(iter_num)/len(iter_num),
                 "run_time":np.mean(runtime)}
    if shift == "linear":
        result_condis[0] = tmp
    elif shift =="abrupt":
        result_condis[1]=tmp
    file_name = 'result.dat'
    with open(file_name, 'wb') as f:
        pickle.dump(tmp, f)

for shift in ["linear", "abrupt"]:
    setpath(trans_setting,example=example, shift=shift)
    with open('result.dat', 'rb') as f:
        t = pickle.load(f)
        t['example']=example
        t['shift']=shift
        if shift == "linear":
            result_condis[0] = t
        elif shift =="abrupt":
            result_condis[1]= t
#%% excel table generator
res = [{} for i in range(2*2)]
for example in ["marginal_paperthreshold", "cdist2"]:
    for shift in ["linear", "abrupt"]:
        setpath(trans_setting,example=example, shift=shift)
        changepoint_err_list = []
        cluster_err_list = []
        iter_num = []
        runtime = []
        for seed in range(M):
            file_name = "seed_"+str(seed)+".dat"
            pkl_file = open(file_name, 'rb')
            t = pickle.load(pkl_file)
            changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                    t['changepoint'].squeeze(), N, T)
            changepoint_err_list.append(changepoint_err)
            cluster_err_list.append(cluster_err)
            iter_num.append(t[0])
            runtime.append(t[2])
            pkl_file.close()
        tmp = {"example":example, "shift":shift,
               "changepoint_err":np.mean(changepoint_err_list),
               "ars":np.mean(cluster_err_list),
               "iter_num":np.mean(iter_num),
               "runtime":np.mean(runtime)}
        
res = result_marginal+result_condis
dat = pd.DataFrame(res)
dat['run_time'] = dat['run_time']/np.timedelta64(1, 's')
dat.to_excel("result1.xlsx")  
res2 = pd.DataFrame(result_mean)
res2['run_time']=res2['run_time']/np.timedelta64(1, 's')
res2.to_excel('tmp.xlsx')
