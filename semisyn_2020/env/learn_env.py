# -*- coding: utf-8 -*-
"""
Learn transition and reward functions from IHS 2020

Created on Fri Jul 26 10:50:17 2024

@author: Liyuan
"""

import sys, os, pickle
# import dill as pickle
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the "ClusterRL" directory
dirl_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Append the ClusterRL directory to sys.path
sys.path.append(dirl_dir)

# semisyn_path =  os.path.abspath(os.path.join(script_dir, '..'))

import numpy as np
import random
from datetime import datetime
from copy import copy, deepcopy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import functions.simu_mean_detect as mean_detect
import functions.compute_test_statistics_separateA as stat
from functions.evaluation_separateA import fitted_Q_evaluation
from functions.evaluation import select_model_cv
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
path_name = os.getcwd()
#%%
# argv should be: seed, kappa, degree, num_threads
seed = int(sys.argv[1])
state_transform = int(sys.argv[2])
method = sys.argv[3]
nthread = int(sys.argv[4])
refit = int(sys.argv[5])
early_stopping = int(sys.argv[6])
max_iter=int(sys.argv[7])
threshold_type=sys.argv[8]
training_phrase=sys.argv[9]
T_train=int(sys.argv[10])
ic_T_dynamic=int(sys.argv[11])
test_cluster_type = sys.argv[12]
is_cv =int(sys.argv[13])
C=int(sys.argv[14]) if sys.argv[14]=="1" else float(sys.argv[14]) 
rbf = False
np.random.seed(seed)
random.seed(seed)

epsilon = 1/T_train
B = 5000
kappa_min = int(0.1*T_train)
kappa_max = T_train
Kl_fun="Nlog(NT)/T"


    
K_list = list(range(1,5))
gamma=0.9
p_var = 3
#%%
def setpath(method, change_dir=1):
    if method in ["proposed", "only_clusters"]:
        Kl_fun_name = "Nlog(NT)_T"
        method_name = method + str(K_list)+'/Kl_fun'+Kl_fun_name+'_C'+str(C)
        if ic_T_dynamic:
            method_name += "_Tdynamic"
        if method == "proposed":
            method_name += "earlystop"+str(early_stopping)+'/max_iter'+str(max_iter)
        print(method_name)
        sys.stdout.flush()
    else:
        method_name  = method
   
    path_name = 'results'+'/T_train'+str(T_train)+'/threshold'+threshold_type+\
        '/'+method_name+'/kappa_min'+str(kappa_min)+'_max'+str(kappa_max)+"/seed"+str(seed)

    
    if change_dir: 
        if not os.path.exists('results'):
            os.makedirs('results', exist_ok=True)
        if not os.path.exists(path_name):
            os.makedirs(path_name, exist_ok=True)
        os.chdir(path_name)
        print(os.getcwd())
        sys.stdout.flush()
    return path_name

# %% read in and process data
def process_data():
    # Read the CSV file
    print('data file', path_name+"/ihs2020_imputed_data_rhr.csv")
    data_path=path_name+"/ihs2020_imputed_data_rhr.csv"
    dat = pd.read_csv(data_path)
    # normalize
    def transform(x):
        return (x - np.mean(x)) / np.std(x)#x/np.std(x)  
    if state_transform:
        dat['STEP_COUNT'] = transform(dat.STEP_COUNT)
        dat['SLEEP_COUNT'] = transform(dat.SLEEP_COUNT)
        dat['MOOD'] = transform(dat.MOOD)
    
    N = len(set(dat.STUDY_PRTCPT_ID))
    T = len(set(dat.Date)) - 1
    # num_weeks = list(set(dat.Week))[-1]
    init_date = list(set(dat.Date))[0]
    # state includes
    States = np.zeros(shape=(N, T + 1, 3))
    # this week's average step count
    States[:, :, 0] = np.array(dat.STEP_COUNT).reshape(N, T + 1)
    # this week's average sleep
    States[:, :, 1] = np.array(dat.SLEEP_COUNT).reshape(N, T + 1)
    # this week's average mood score
    States[:, :, 2] = np.array(dat.MOOD).reshape(N, T + 1)
    # # previous week's average step count
    # States[:, 1:, 3] = np.array(dat.STEP_COUNT[dat["Week"] != num_weeks]).reshape(N, T)
    # States[:, 0, 3] = np.array(dat.weekly_step_t[dat["study_week"] == 0])
    Rewards = np.array(dat.STEP_COUNT[dat["Day"] != 0]).reshape(N, T)
    # action is activity message vs all the rest
    dat['action_step_vs_other'] = (dat['Notification_type'] == "steps").astype(int)
    Actions = np.array(dat.action_step_vs_other[dat["Day"] != 0]).reshape(N, T)
    
    specialty = dat.TeamName[dat["Date"] == init_date].str.split('_').str[1]
    return States, Actions, Rewards, specialty
#%%
def estimate_env_for_each_block(States, Actions, Rewards, file_name):
    if os.path.exists(file_name) and not refit:
        with open(file_name, "rb") as f:
            res = pickle.load(f)
        changepoints = res["changepoints"]
        g_index = res["clustering"]
        
        with open('env/proposed_g0t0.dat', "rb") as f:
            res = pickle.load(f)
        proposed_g0t0_changepoints = res["changepoints"]
        proposed_g0t0_g_index = res["clustering"]
        
        with open('env/proposed_g1t0.dat', "rb") as f:
            res = pickle.load(f)
        proposed_g1t0_changepoints = res["changepoints"]
        proposed_g1t0_g_index = res["clustering"]
        split_cluster = {
            0:{'changepoints':proposed_g0t0_changepoints,
                            "g_index":proposed_g0t0_g_index
                            },
                         1:{'changepoints':proposed_g1t0_changepoints,
                            "g_index":proposed_g1t0_g_index
                            }
                         }
    else:
        raise ValueError("no change point and clusters fitted.")
    
    re_transition_dict = {g:None for g in range(len(set(g_index))*2+1)}
    re_reward_dict = {g:None for g in range(len(set(g_index))*2+1)}
    # c=0
    # for g in np.unique(g_index):
    #     print('g',g)
    #     for t in range(2):
    #         print('t', t)
    #         if t == 0:
    #             s = States[g_index == g, : changepoints[np.where(g_index==g)[0][0]], :].reshape((-1, p_var))
    #             ns = States[g_index == g, 1 : changepoints[np.where(g_index==g)[0][0]]+1, :].reshape((-1, p_var))
    #             r = Rewards[g_index == g, : changepoints[np.where(g_index==g)[0][0]]].reshape(-1, 1)
    #             a = Actions[g_index == g, : changepoints[np.where(g_index==g)[0][0]]].reshape(-1, 1)
                
    #         else:
    #             s = States[g_index == g, changepoints[np.where(g_index==g)[0][0]]:-1, :].reshape((-1, p_var))
    #             ns = States[g_index == g, changepoints[np.where(g_index==g)[0][0]]+1:, :].reshape((-1, p_var))
    #             r = Rewards[g_index == g, changepoints[np.where(g_index==g)[0][0]]: ].reshape(-1, 1)
    #             a = Actions[g_index == g, changepoints[np.where(g_index==g)[0][0]]: ].reshape(-1, 1)
            
    #         poly = PolynomialFeatures(2, interaction_only=True)
    #         X = poly.fit_transform(np.hstack((a,s)))
    #         X = np.delete(X, np.s_[2 + p_var*2:X.shape[1]], 1)
    #         # X = np.concatenate([np.ones((s.shape[0],1)), s, s*a], axis=-1)
    #         transition_dict[c] = LinearRegression(fit_intercept=False).fit(X, ns).coef_
    #         reward_dict[c] = LinearRegression(fit_intercept=False).fit(X, r).coef_
    #         c += 1
    #         print('c', c)
    
    # Fit the reference class (g=0, t=0)
    # g_ref = 0
    # t_ref = 0
    beta_0  ={"beta_0_transition":None,
              "beta_0_reward":None}
    # beta_0_transition=None
    # beta_0_reward=None
    def fit_model(a,s,r, c):
        poly = PolynomialFeatures(2, interaction_only=True)
        X = poly.fit_transform(np.hstack((a, s)))
        X = np.delete(X, np.s_[2 + p_var * 2:X.shape[1]], 1)

        if c==0:
            # Fit the reference class coefficients
            beta_0["beta_0_transition"] = LinearRegression(fit_intercept=False).fit(X, ns).coef_
            beta_0["beta_0_reward"] = LinearRegression(fit_intercept=False).fit(X, r).coef_
            re_transition_dict[c] = np.zeros(beta_0["beta_0_transition"].shape)
            re_reward_dict[c] = np.zeros(beta_0["beta_0_reward"].shape)
        else:
            # Fit subsequent classes with reference to beta_0
            transition_model = LinearRegression(fit_intercept=False)
            reward_model = LinearRegression(fit_intercept=False)
            transition_coef = transition_model.fit(X, ns - X @ beta_0["beta_0_transition"].T).coef_ #+ beta_0_transition
            reward_coef = reward_model.fit(X, r - X @ beta_0["beta_0_reward"].T).coef_ #+ beta_0_reward
            re_transition_dict[c] = transition_coef
            re_reward_dict[c] = reward_coef
            
    c = 0
    split_g=1
    for g in np.unique(g_index):
        for t in range(2):
            if t == 0:
                if g==split_g:
                    split_changepoints = split_cluster[g]["changepoints"]
                    for tt in range(2):
                        if tt==0:
                            s = States[g_index == g, :split_changepoints[0], :].reshape((-1, p_var))
                            ns = States[g_index == g, 1:split_changepoints[0] + 1, :].reshape((-1, p_var))
                            r = Rewards[g_index == g, :split_changepoints[0]].reshape(-1, 1)
                            a = Actions[g_index == g, :split_changepoints[0]].reshape(-1, 1)
                        else:
                            s = States[g_index == g, split_changepoints[0]:changepoints[np.where(g_index == g)[0][0]], :].reshape((-1, p_var))
                            ns = States[g_index == g, 1+split_changepoints[0]:changepoints[np.where(g_index == g)[0][0]]+1, :].reshape((-1, p_var))
                            r = Rewards[g_index == g, split_changepoints[0]:changepoints[np.where(g_index == g)[0][0]]].reshape(-1, 1)
                            a = Actions[g_index == g, split_changepoints[0]:changepoints[np.where(g_index == g)[0][0]]].reshape(-1, 1)
                        fit_model(a, s, r, c)
                        c += 1
                else:
                
                    s = States[g_index == g, :changepoints[np.where(g_index == g)[0][0]], :].reshape((-1, p_var))
                    ns = States[g_index == g, 1:changepoints[np.where(g_index == g)[0][0]] + 1, :].reshape((-1, p_var))
                    r = Rewards[g_index == g, :changepoints[np.where(g_index == g)[0][0]]].reshape(-1, 1)
                    a = Actions[g_index == g, :changepoints[np.where(g_index == g)[0][0]]].reshape(-1, 1)
                    fit_model(a, s, r, c)
                    c += 1
            else:
                s = States[g_index == g, changepoints[np.where(g_index == g)[0][0]]:-1, :].reshape((-1, p_var))
                ns = States[g_index == g, changepoints[np.where(g_index == g)[0][0]] + 1:, :].reshape((-1, p_var))
                r = Rewards[g_index == g, changepoints[np.where(g_index == g)[0][0]]:].reshape(-1, 1)
                a = Actions[g_index == g, changepoints[np.where(g_index == g)[0][0]]:].reshape(-1, 1)
                fit_model(a, s, r, c)
                c += 1
                
            print('c', c)
    #%%
    # re_transition_dict[4]=np.random.normal(scale=0.1, size=re_transition_dict[0].shape)
    re_transition_dict[4]=np.random.uniform(low=-0.25,high=0.25, size=re_transition_dict[0].shape)

    diff = {}
    for key in re_transition_dict.keys():
        for key2 in re_transition_dict.keys():
            if key2>key:
                diff_val=np.linalg.norm( re_transition_dict[key] -  re_transition_dict[key2])
                print(key, "-", key2, diff_val)
                diff[(key, key2)] = diff_val
    
    print({k:v for k,v in sorted(diff.items(), key=lambda item: -1*item[1])})
    if diff[(1,4)]>0.6 and diff[(3,4)]>0.6:
        print('888888')
    env['transition']  =re_transition_dict
    with open('env/re_learnt_env_randomstationary.dat', "wb") as f:
        pickle.dump(env, f)

    #%%
    with open('env/re_learnt_env_randomstationary.dat', "wb") as f:
        pickle.dump({"transition": re_transition_dict,
                     "reward": re_reward_dict,
                     "beta_0_transition":beta_0["beta_0_transition"],
                     "beta_0_reward":beta_0["beta_0_reward"]}, f)


    with open('env/re_learnt_env_split_secondcluster.dat', "wb") as f:
        pickle.dump({"transition": re_transition_dict,
                     "reward": re_reward_dict,
                     "beta_0_transition":beta_0["beta_0_transition"],
                     "beta_0_reward":beta_0["beta_0_reward"]}, f)
#%% t test on means before and after change points
import scipy.stats as stats
import pingouin as pg
from statsmodels.multivariate.manova import MANOVA
States, Actions, Rewards, specialty = process_data()
with open('D:/OneDrive/PhD/DIRL/IHS/simu/simu_anonymous/semisyn_2020/env/proposed.dat', "rb") as f:
    res = pickle.load(f)
changepoints = res["changepoints"]
g_index = res["clustering"]

## proposed method
# cluster 1
g = 0
s_before = States[g_index == g, :changepoints[np.where(g_index == g)[0][0]], :].reshape((-1, p_var))
s_after = States[g_index == g, changepoints[np.where(g_index == g)[0][0]]:, :].reshape((-1, p_var))
# Combine data into a pandas DataFrame
data = np.vstack([s_before,s_after])
labels = np.array([1]*s_before.shape[0] + [2]*s_after.shape[0])

df = pd.DataFrame(data, columns=['Dim1', 'Dim2', 'Dim3'])
df['Group'] = labels

# Perform MANOVA
maov = MANOVA.from_formula('Dim1 + Dim2 + Dim3 ~ Group', data=df)
print(maov.mv_test())
#                    Multivariate linear model
# ===============================================================
                                                               
# ---------------------------------------------------------------
#        Intercept        Value  Num DF   Den DF   F Value Pr > F
# ---------------------------------------------------------------
#           Wilks' lambda 0.9989 3.0000 43676.0000 15.8477 0.0000
#          Pillai's trace 0.0011 3.0000 43676.0000 15.8477 0.0000
#  Hotelling-Lawley trace 0.0011 3.0000 43676.0000 15.8477 0.0000
#     Roy's greatest root 0.0011 3.0000 43676.0000 15.8477 0.0000
# ---------------------------------------------------------------
                                                               
# ---------------------------------------------------------------
#          Group          Value  Num DF   Den DF   F Value Pr > F
# ---------------------------------------------------------------
#           Wilks' lambda 0.9973 3.0000 43676.0000 39.4211 0.0000
#          Pillai's trace 0.0027 3.0000 43676.0000 39.4211 0.0000
#  Hotelling-Lawley trace 0.0027 3.0000 43676.0000 39.4211 0.0000
#     Roy's greatest root 0.0027 3.0000 43676.0000 39.4211 0.0000
# ===============================================================

# cluster 2
g = 1
s_before = States[g_index == g, :changepoints[np.where(g_index == g)[0][0]], :].reshape((-1, p_var))
s_after = States[g_index == g, changepoints[np.where(g_index == g)[0][0]]:, :].reshape((-1, p_var))
# Combine data into a pandas DataFrame
data = np.vstack([s_before,s_after])
labels = np.array([1]*s_before.shape[0] + [2]*s_after.shape[0])

df = pd.DataFrame(data, columns=['Dim1', 'Dim2', 'Dim3'])
df['Group'] = labels

# Perform MANOVA
maov = MANOVA.from_formula('Dim1 + Dim2 + Dim3 ~ Group', data=df)
print(maov.mv_test())
#                    Multivariate linear model
# ===============================================================
                                                               
# ---------------------------------------------------------------
#        Intercept        Value  Num DF   Den DF   F Value Pr > F
# ---------------------------------------------------------------
#           Wilks' lambda 0.9988 3.0000 42332.0000 16.9924 0.0000
#          Pillai's trace 0.0012 3.0000 42332.0000 16.9924 0.0000
#  Hotelling-Lawley trace 0.0012 3.0000 42332.0000 16.9924 0.0000
#     Roy's greatest root 0.0012 3.0000 42332.0000 16.9924 0.0000
# ---------------------------------------------------------------
                                                               
# ---------------------------------------------------------------
#          Group          Value  Num DF   Den DF   F Value Pr > F
# ---------------------------------------------------------------
#           Wilks' lambda 0.9945 3.0000 42332.0000 77.6455 0.0000
#          Pillai's trace 0.0055 3.0000 42332.0000 77.6455 0.0000
#  Hotelling-Lawley trace 0.0055 3.0000 42332.0000 77.6455 0.0000
#     Roy's greatest root 0.0055 3.0000 42332.0000 77.6455 0.0000
# ===============================================================

## homogenous method
s_before = States[:, :74, :].reshape((-1, p_var))
s_after = States[:,74:, :].reshape((-1, p_var))
data = np.vstack([s_before,s_after])
labels = np.array([1]*s_before.shape[0] + [2]*s_after.shape[0])

df = pd.DataFrame(data, columns=['Dim1', 'Dim2', 'Dim3'])
df['Group'] = labels
# Perform MANOVA
maov = MANOVA.from_formula('Dim1 + Dim2 + Dim3 ~ Group', data=df)
print(maov.mv_test())
#                    Multivariate linear model
# ===============================================================
                                                               
# ---------------------------------------------------------------
#        Intercept        Value  Num DF   Den DF   F Value Pr > F
# ---------------------------------------------------------------
#           Wilks' lambda 0.9997 3.0000 86012.0000  7.2169 0.0001
#          Pillai's trace 0.0003 3.0000 86012.0000  7.2169 0.0001
#  Hotelling-Lawley trace 0.0003 3.0000 86012.0000  7.2169 0.0001
#     Roy's greatest root 0.0003 3.0000 86012.0000  7.2169 0.0001
# ---------------------------------------------------------------
                                                               
# ---------------------------------------------------------------
#          Group          Value  Num DF   Den DF   F Value Pr > F
# ---------------------------------------------------------------
#           Wilks' lambda 0.9997 3.0000 86012.0000  7.8213 0.0000
#          Pillai's trace 0.0003 3.0000 86012.0000  7.8213 0.0000
#  Hotelling-Lawley trace 0.0003 3.0000 86012.0000  7.8213 0.0000
#     Roy's greatest root 0.0003 3.0000 86012.0000  7.8213 0.0000
# ===============================================================

#%%
def main():
    States, Actions, Rewards, specialty = process_data()
    estimate_env_for_each_block(States, Actions, Rewards, file_name='proposed.dat')
    