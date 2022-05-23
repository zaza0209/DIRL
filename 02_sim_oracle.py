import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/results")
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") # 引用模块的地址
import simu.compute_test_statistics as stat
import simu.simulate_data_1d as sim
import simu.simu_mean_detect as mean_detect
seed = 1
gamma = 0.9
trans_setting = 'pwconst2'
reward_setting = 'homo'
N = int(40)

np.random.seed(seed)

# %% simulate data
# terminal timestamp
T = 200
# dimension of X0
p = 1
# mean vector of X0
mean0 = 0
# diagonal covariance of X0
cov0 = 0.5
# mean vector of random errors zt
mean = 0
# diagonal covariance of random errors zt
cov = 0

# width of smooth transition function
w = 0.01
delta = 1/10

# the location of change point
# changepoint_list = [int(T/2) + 10, int(T/2) - 10]
changepoint_list = [T, 0]
N = 10
States = np.zeros([N, T, p])
Rewards = np.zeros([N, T-1])
Actions = np.zeros([N, T-1])
signal = 0.5
for changepoint in changepoint_list:
    sim_dat = sim.simulate_data(int(N/2), T, changepoint+1, delta)
    if trans_setting == 'pwconst2' and reward_setting == 'homo':
        def mytransition_function(t):
            return sim_dat.transition_pwconstant2(t, mean, cov, signal=signal)
        def myreward_function(t):
            return sim_dat.reward_homo()
        States0, Rewards0, Actions0 = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed + changepoint_list.index(changepoint))
    elif trans_setting == 'smooth' and reward_setting == 'homo':
        def mytransition_function(t):
            return sim_dat.transition_smooth2(t, mean, cov, w)
        def myreward_function(t):
            return sim_dat.reward_homo()
        States0, Rewards0, Actions0 = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function, seed + changepoint_list.index(changepoint))
    States[changepoint_list.index(changepoint)*int(N/2)+ 0:changepoint_list.index(changepoint)*int(N/2)+int(N/2), :, :] = States0
    Rewards[changepoint_list.index(changepoint)*int(N/2)+ 0:changepoint_list.index(changepoint)*int(N/2)+int(N/2), :] = Rewards0
    Actions[changepoint_list.index(changepoint)*int(N/2)+ 0:changepoint_list.index(changepoint)*int(N/2)+int(N/2), :] = Actions0

# normalize state variables
def transform(x):
    return (x - np.mean(x)) / np.std(x)
for i in range(1):
    States[:,:,i] = transform(States[:,:,i])
# %% Case 1 oracle: run Q learning with rbf basis function approximation
kappa = T

np.random.seed(seed)
T_total = T
startTime = datetime.now()
out_changepoint = mean_detect.detect_mean(States, init = "changepoints", C1=1, C2=1/2)
print('Finished. Time: ', datetime.now() - startTime)

startTime = datetime.now()
out_cluster = mean_detect.detect_mean(States, init = "clusters", C1=1, C2=1/2)
print('Finished. Time: ', datetime.now() - startTime)
