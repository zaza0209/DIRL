# -*- coding: utf-8 -*-
"""
IHS2020

Created on Thu Jun 13 08:33:37 2024

@author: Liyuan
"""
import sys, os, pickle
# import dill as pickle
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the "ClusterRL" directory
dirl_dir = os.path.abspath(os.path.join(script_dir, '..'))

# Append the ClusterRL directory to sys.path
sys.path.append(dirl_dir)

import numpy as np
import random
from datetime import datetime
from copy import copy, deepcopy
import functions.simu_mean_detect as mean_detect
import functions.compute_test_statistics_separateA as stat
from functions.evaluation_separateA import fitted_Q_evaluation
from functions.evaluation import select_model_cv
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
path_name = os.getcwd()
#%%
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
is_r_orignial_scale=int(sys.argv[15])
rbf = False
np.random.seed(seed)
random.seed(seed)

epsilon = 1/T_train
B = 10000
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
def transform_back(x):
    return (x*4.141009340555169 + 19.343480280896884)**3

def process_data():
    # Read the CSV file
    print('data file', path_name+"/ihs2020_imputed_data_rhr.csv")
    # dat=pd.read_csv('D:/OneDrive/PhD/DIRL/IHS/simu/simu_anonymous/realdata_2020/ihs2020_imputed_data_rhr.csv')
    dat = pd.read_csv(path_name+"/ihs2020_imputed_data_rhr.csv")
    N = len(set(dat.STUDY_PRTCPT_ID))
    T = len(set(dat.Date)) - 1
    # normalize
    def transform(x):
        return (x - np.mean(x)) / np.std(x) 
    States_original = np.zeros((N, T+1, 3))
    States_original[:, :, 0] = np.array(dat.STEP_COUNT**3).reshape(N, T + 1)
    # this week's average sleep
    States_original[:, :, 1] = np.array(dat.SLEEP_COUNT**2).reshape(N, T + 1)
    # this week's average mood score
    States_original[:, :, 2] = np.array(dat.MOOD).reshape(N, T + 1)
    
    if state_transform:
        # mean 19.343480280896884, std: 4.141009340555169
        dat['STEP_COUNT'] = transform(dat.STEP_COUNT)
        dat['SLEEP_COUNT'] = transform(dat.SLEEP_COUNT)
        dat['MOOD'] = transform(dat.MOOD)
    
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
    # with open("olddata.dat", "wb") as f:
    #     pickle.dump({'States':States,
    #                  'Actions':Actions, 'Rewards':Rewards}, f)  
    # print(Actions[0])
    return States, Actions, Rewards, specialty, States_original 

#%% detect change points and clusterings
def cp_cluster_detection(States_train, Actions_train, Rewards_train, file_name):
    if os.path.exists(file_name) and not refit:
        with open(file_name, "rb") as f:
            res = pickle.load(f)
        changepoints = res["changepoints"]
        g_index = res["clustering"]
        return changepoints, g_index 
    
    N = States_train.shape[0]
    T = Actions_train.shape[1]
    
    if method == "overall":
        g_index = np.repeat(0, N)
        changepoints = np.repeat(0, N)
        
    elif method == "only_clusters":
        changepoints = np.repeat(0, N)
        out = mean_detect.fit(States_train, Actions_train, init = "changepoints",
                              nthread=nthread, changepoints_init = changepoints, 
                              max_iter=1, is_only_cluster=1, C=C, 
                              K = K_list,
                              Kl_fun=Kl_fun, ic_T_dynamic=ic_T_dynamic)
        g_index = out.g_index
        loss = out.loss
        
    elif method == "only_cp":
        g_index = np.repeat(0, N)
        out = mean_detect.fit(States_train, Actions_train, example="cdist", seed = seed,
                              init = "clustering", B=B,
                              epsilon=epsilon, nthread=nthread,
                              kappa_min = kappa_min, kappa_max = kappa_max,
                              max_iter=1, alpha=0.01,
                              g_index_init=g_index,
                              is_only_cp=1,
                              K=K_list, init_cluster_range = T - 1 - kappa_min,
                              Kl_fun=Kl_fun, C=C, threshold_type = threshold_type)
        changepoints = out.changepoints
        loss = out.loss
        
    elif method == "proposed":
        import time
        start_time = time.time()
        out = mean_detect.fit(States_train, Actions_train, example="cdist", seed = seed,
                              init = "clustering",  B=B,
                              epsilon=epsilon,  nthread=nthread,
                              kappa_min = kappa_min, kappa_max = kappa_max,
                              max_iter=max_iter,alpha=0.01,
                              K=K_list, init_cluster_range = T - 1 - kappa_min,
                              Kl_fun=Kl_fun, C=C, threshold_type = threshold_type,
                              early_stopping=early_stopping, ic_T_dynamic=ic_T_dynamic)
        end_time = time.time()
        print(end_time - start_time)
        g_index = out.g_index
        changepoints = out.changepoints
        loss = out.loss
    else:
        raise ValueError("Invalid method")
    if method != "overall":
        with open(file_name, "wb") as f:
            pickle.dump({'changepoints':changepoints, 'clustering':g_index, 
                         'loss':loss,
                         'iter_num':out.iter_num,
                          'K_path':out.K_path
                         }, f)
    else:
        with open(file_name, "wb") as f:
            pickle.dump({'changepoints':changepoints, 'clustering':g_index, 
                         }, f)
    return changepoints, g_index

#%% policy evaluation
def evaluate(States, Actions, Rewards, changepoints, g_index):
    # number of training time points
    #%%
    N=States.shape[0]
    States_train = copy(States[:, :T_train+1, :])
    Actions_train = copy(Actions[:, :T_train])
    Rewards_train = copy(Rewards[:, :T_train])
    States_test = copy(States[:, T_train:T_train+2, :]).reshape(N, 2, -1)
    Actions_test = copy(Actions[:, T_train]).reshape(N, 1)
    Rewards_test = copy(Rewards[:, T_train]).reshape(N, 1)
    
    if is_r_orignial_scale:
        Rewards_train = transform_back(Rewards_train)
        Rewards_test = transform_back(Rewards_test)
    #%% estimate the optimal policy
    rbf_bw = 0.1
    qmodel = 'polynomial'
    degree=1
    metric = 'ls'
    basemodel = DecisionTreeRegressor(random_state=seed)
    q_function_list_group =  [None]*len(np.unique(g_index))
        
    for g in np.unique(g_index):
        print('g', g)
        if is_r_orignial_scale:
            q_function_file ='q_function_list_'+str(g)+"_origianlR.pkl"
        else:
            q_function_file ='q_function_list_'+str(g)+".pkl"
        if os.path.exists(q_function_file) and not refit:
            with open(q_function_file, "rb") as f:
                tmp = pickle.load(f)
            q_function_list_group[g] = tmp
            
        else:
            if is_cv:
                param_grid = {"max_depth": [3, 4, 5], "min_samples_leaf": [40, 50, 60]}
                print('States_train[g_index == ',g,', ', changepoints[np.where(g_index==g)[0][0]], ':, :]',States_train[g_index == g, changepoints[np.where(g_index==g)[0][0]]:, :].shape)
                out = select_model_cv(States_train[g_index == g, changepoints[np.where(g_index==g)[0][0]]:, :].reshape((np.sum(g_index == g), -1, p_var)), 
                                      Rewards_train[g_index == g, changepoints[np.where(g_index==g)[0][0]]:].reshape((np.sum(g_index == g), -1)),
                                      Actions_train[g_index == g, changepoints[np.where(g_index==g)[0][0]]:].reshape((np.sum(g_index == g), -1)), 
                                      param_grid, bandwidth=rbf_bw,
                                      qmodel='polynomial', gamma=gamma, model=basemodel, max_iter=200, tol=1e-4,
                                      nfold = 5, num_threads = nthread, metric = metric)
                model = out['best_model']
            else:
                print('not cv')
                basemodel.set_params(max_depth=3, min_samples_leaf=50)
                model = copy(basemodel)
            q_all_group =stat.q_learning(States_train[g_index == g, changepoints[np.where(g_index==g)[0][0]]:, :],
                                            Rewards_train[g_index == g, changepoints[np.where(g_index==g)[0][0]]:], 
                                            Actions_train[g_index == g, changepoints[np.where(g_index==g)[0][0]]:], 
                                            qmodel, degree, gamma, rbf_dim=degree, rbf_bw=rbf_bw)
            q_all_fit = q_all_group.fit(model, max_iter=200, tol = 1e-6, 
                                        early_stopping=early_stopping)
            q_function_list_group[g] = q_all_fit.q_function_list
            with open(q_function_file, "wb") as f:
                pickle.dump(q_function_list_group[g], f)
            print('saved', q_function_file)

    #%% next evaluation using the offline data
    if test_cluster_type == "test_set_only":
        save_path = path_name +'/results/T_train'+str(T_train)+'/test_g_index_Tdynamic'+str(ic_T_dynamic)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(path_name +'/results/T_train'+str(T_train)+'/test_g_index_Tdynamic'+str(ic_T_dynamic))
            sys.stdout.flush()
        test_cluster_file = save_path+'/seed'+str(seed)+'.dat'
    elif test_cluster_type == "after_cp":
        test_cluster_file = 'test_g.dat'
    elif test_cluster_type == "after_proposed_cp":
        proposed_path = path_name+'/'+setpath("proposed", change_dir=0)
        test_cluster_file = proposed_path+ '/after_proposed_cp/test_g.dat'
    elif test_cluster_type == "proposed_g_index":
        proposed_path = path_name+'/'+setpath("proposed", change_dir=0)
        test_cluster_file = proposed_path+ '/proposed.dat'
    elif test_cluster_type == "g_index":
        proposed_path = path_name+'/'+setpath(method, change_dir=0)
        test_cluster_file = proposed_path+"/"+ method+'.dat'
    else:
        raise ValueError("Invalid test_cluster_type.")
        
    
    if test_cluster_type == "proposed_g_index" and os.path.exists(test_cluster_file):
        check_file=1
    elif test_cluster_type != "proposed_g_index" and not refit:
        check_file=1
    else:
        check_file=0
    if check_file:
        with open(test_cluster_file, 'rb') as f:
            clusters_test = pickle.load(f)
        if test_cluster_type[-7:] == "g_index":
            clusters_test = clusters_test["clustering"]
    else:
        if test_cluster_type == "test_set_only":
            out = mean_detect.fit(States_test, Actions_test, init = "changepoints",
                                  nthread=nthread, changepoints_init = np.repeat(0, N), 
                                  max_iter=1, is_only_cluster=1, C=C, 
                                  K = K_list,
                                  Kl_fun=Kl_fun)
            # clusters_test =out.g_index
            
        elif test_cluster_type == "after_cp":
            out = mean_detect.fit(States, Actions, init = "changepoints",
                                  nthread=nthread, changepoints_init = changepoints, 
                                  max_iter=1, is_only_cluster=1, C=C, 
                                  K = K_list,
                                  Kl_fun=Kl_fun, ic_T_dynamic=ic_T_dynamic)
            # clusters_test =out.g_index
            
        elif test_cluster_type == "after_proposed_cp":
            proposed_cp_path = proposed_path+ '/proposed.dat'
            with open(proposed_cp_path, "rb") as f:
                res = pickle.load(f)
            changepoints_proposed = res['changepoints']
            out = mean_detect.fit(States, Actions, init = "changepoints",
                                  nthread=nthread,
                                  changepoints_init = changepoints_proposed, 
                                  max_iter=1, is_only_cluster=1, C=C, 
                                  K = K_list,
                                  Kl_fun=Kl_fun, ic_T_dynamic=ic_T_dynamic)
        clusters_test =out.g_index
        
        with open(test_cluster_file, "wb") as f:
            pickle.dump(clusters_test, f)
    
    print(pd.crosstab(g_index,
                      clusters_test, rownames=['g_index'], colnames=['Cluster_test'])
          )
    df_cluster_rewards = pd.DataFrame(columns=['id', 'method', 'cluster_train', 'cluster_test', 'changepoint', 'optimal_reward'])
    df_cluster_rewards['id'] = np.arange(States.shape[0])
    df_cluster_rewards['method'] = method
    df_cluster_rewards['cluster_id'] = g_index
    df_cluster_rewards['cluster_test'] = clusters_test
    df_cluster_rewards['changepoint'] = changepoints
    df_cluster_rewards['optimal_reward'] = 0.0
    
    #%%
    for k in np.unique(clusters_test):
        for k_train in np.unique(g_index):
            indices = np.where((clusters_test == k) & (g_index == k_train))[0]
            print(len(indices))
            if len(indices)==0:
                continue
            
            States_k_ktrain = States_test[indices, :, :]
            Rewards_k_ktrain = Rewards_test[indices, :]
            Actions_k_ktrain = Actions_test[indices, :]
            q_all_eval = stat.q_learning(States_k_ktrain,
                                         Rewards_k_ktrain,
                                         Actions_k_ktrain, qmodel, degree, gamma)
            
            if is_r_orignial_scale:
                ope_file_name ="ope_q_k"+str(k)+"_ktrain"+str(k_train)+"_origianlR.pkl"
            else:
                ope_file_name ="ope_q_k"+str(k)+"_ktrain"+str(k_train)+".pkl"
                
            if os.path.exists(ope_file_name) and not refit:
                with open(ope_file_name, "rb") as f:
                    tmp = pickle.load(f)
                q_all_eval.q_function_list = tmp
                estimated_value_optimal = fitted_Q_evaluation(qlearn_env=q_all_eval,
                                                              max_iter=0)
                
            else:
                q_all_eval.q_function_list = deepcopy(q_function_list_group[k_train])
                # estimate the optimal values for each individual
                estimated_value_optimal = fitted_Q_evaluation(qlearn_env=q_all_eval,
                                                              max_iter=500, 
                                                              early_stopping=early_stopping)
                
                with open(ope_file_name, "wb") as f:
                    pickle.dump(q_all_eval.q_function_list, f)
                    
            print("k =", k, ", k_train =", k_train, ", values_k =", np.mean(estimated_value_optimal))
            df_cluster_rewards.loc[indices, 'optimal_reward'] = estimated_value_optimal

#%%
    ope_value = df_cluster_rewards['optimal_reward'].mean()
    print('ope_value',ope_value)
    sys.stdout.flush()
    
    return ope_value, df_cluster_rewards


#%% main
def main():
    _ = setpath(method)
    if test_cluster_type == "proposed_g_index" and method != "proposed":
        test_cluster_type_name = test_cluster_type + "earlystop"+str(early_stopping)+'/max_iter'+str(max_iter)
    else:
        test_cluster_type_name = test_cluster_type + "earlystop"+str(early_stopping)
    ope_path = os.getcwd()+'/'+test_cluster_type_name+'/is_cv'+str(is_cv)+'_Tdynamic'+str(ic_T_dynamic)
    print(ope_path)
    sys.stdout.flush()
    
    if not os.path.exists(ope_path):
        os.makedirs(ope_path)
    if is_r_orignial_scale:
        file_name = ope_path+"/"+"seed_origianlR.dat"
    else:
        file_name = ope_path+"/"+"seed.dat"
    if os.path.exists(file_name) and not refit and training_phrase != "plot":
        exit()
        
    # direct the screen output to a file
    stdoutOrigin = sys.stdout
    if not refit:
        sys.stdout = open("log_originalR"+str(is_r_orignial_scale)+".txt", "a")
    else:
        sys.stdout = open("log_originalR"+str(is_r_orignial_scale)+".txt", "w")
    print("\nName of Python script:", sys.argv[0])
    sys.stdout.flush()
    #%% data
    States, Actions, Rewards, specialty, States_original = process_data()
    #%% detect change points and clusters 
    method_file_name = method+'.dat'
    changepoints, g_index = cp_cluster_detection(copy(States[:, :T_train+1, :]),
                                                 copy(Actions[:, :T_train]),
                                                 copy(Rewards[:, :T_train]), 
                                                 method_file_name)
    print('change points', np.unique(changepoints, return_counts=1))
    print("clusters", np.unique(g_index, return_counts=1))
    sys.stdout.flush()
    if training_phrase == "cp_cluster":
        exit()
    #%% cluster result analysis
    import matplotlib.pyplot as plt
    # Variable names
    variables = ['step', 'sleep', 'mood']
    y_label_name = {'step':'Cubic root of step counts',
                    'sleep':'Square root of sleep duration',
                    'mood': 'Mood score'}
    # Plotting
    fig, axes = plt.subplots(3, len(set(g_index)), figsize=(12, 18), sharey='row')
    
    for var_idx, var_name in enumerate(variables):
        for cluster in list(set(g_index)):
            if len(set(g_index))>1:
                ax = axes[var_idx, cluster]
            else:
                ax = axes[var_idx]
            subjects_in_cluster = np.where(g_index == cluster)[0]
    
            for subject in subjects_in_cluster:
                trajectory = States[subject, :, var_idx]
                ax.plot(trajectory, alpha=0.1, color='black')
            
            change_point = changepoints[subject]
            ax.axvline(change_point, color='red', linestyle='--', linewidth=2)
            
            mean_before = np.mean(States[subjects_in_cluster, :change_point, var_idx])
            mean_after = np.mean(States[subjects_in_cluster, change_point:, var_idx])
            
            ax.hlines(mean_before, xmin=0, xmax=change_point, color='green', linestyle='--', linewidth=2)
            ax.hlines(mean_after, xmin=change_point, xmax=States.shape[1], color='yellow', linestyle='--', linewidth=2)

            # ax.set_title(f'Cluster {cluster+1} - {var_name}')
            ax.set_title(f'Cluster {cluster+1}')
            ax.set_xlabel('Time Step')
            if cluster == 0:
                ax.set_ylabel(y_label_name[var_name])
            
    plt.tight_layout()
    #%%
    plt.savefig(method+"_cluster_changepoints.png")
    #%%
    # Loop over each variable
    for var_idx, var_name in enumerate(variables):
        # Create a new figure
        fig, axes = plt.subplots(1, len(set(g_index)), figsize=(12, 6), sharey='row')
        
        # Loop over each cluster
        for cluster in list(set(g_index)):
            if len(set(g_index)) > 1:
                ax = axes[cluster]
            else:
                ax = axes
            
            subjects_in_cluster = np.where(g_index == cluster)[0]
            
            # Plot trajectories
            for subject in subjects_in_cluster:
                trajectory = States[subject, :, var_idx]
                ax.plot(trajectory, alpha=0.1, color='black')
            
            # Plot change point
            change_point = changepoints[subject]
            ax.axvline(change_point, color='red', linestyle='--', linewidth=2)
            
            # Plot means before and after change point
            mean_before = np.mean(States[subjects_in_cluster, :change_point, var_idx])
            mean_after = np.mean(States[subjects_in_cluster, change_point:, var_idx])
            
            ax.hlines(mean_before, xmin=0, xmax=change_point, color='green', linestyle='--', linewidth=2)
            ax.hlines(mean_after, xmin=change_point, xmax=States.shape[1], color='yellow', linestyle='--', linewidth=2)
    
            # ax.set_title(f'Cluster {cluster} - {var_name}')
            ax.set_title(f'Cluster {cluster+1}')
            ax.set_xlabel('Time Step')
            if cluster == 0:
                ax.set_ylabel(y_label_name[var_name])
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{method}_{var_name}_cluster_changepoints.png")
        plt.close(fig)  # Close the figure to free up memory
    #%% plots in original scale
    import matplotlib.pyplot as plt
    # Variable names
    variables = ['step', 'sleep', 'mood']
    y_label_name = {'step':'Step counts',
                    'sleep':'Sleep duration',
                    'mood': 'Mood score'}
    
    # Plotting
    fig, axes = plt.subplots(3, len(set(g_index)), figsize=(12, 18), sharey='row')
    
    for var_idx, var_name in enumerate(variables):
        for cluster in list(set(g_index)):
            if len(set(g_index))>1:
                ax = axes[var_idx, cluster]
            else:
                ax = axes[var_idx]
            subjects_in_cluster = np.where(g_index == cluster)[0]
    
            for subject in subjects_in_cluster:
                trajectory = States_original[subject, :, var_idx]
                ax.plot(trajectory, alpha=0.1, color='black')
            
            change_point = changepoints[subject]
            ax.axvline(change_point, color='red', linestyle='--', linewidth=2)
            
            mean_before = np.mean(States_original[subjects_in_cluster, :change_point, var_idx])
            mean_after = np.mean(States_original[subjects_in_cluster, change_point:, var_idx])
            
            ax.hlines(mean_before, xmin=0, xmax=change_point, color='green', linestyle='--', linewidth=2)
            ax.hlines(mean_after, xmin=change_point, xmax=States.shape[1], color='yellow', linestyle='--', linewidth=2)

            # ax.set_title(f'Cluster {cluster} - {var_name}')
            ax.set_title(f'Cluster {cluster+1}')
            ax.set_xlabel('Time Step')
            if cluster == 0:
                ax.set_ylabel(y_label_name[var_name])
            
    plt.tight_layout()
    #%%
    plt.savefig(method+"_cluster_changepoints_originalscale.png")
    #%%
    # Loop over each variable
    for var_idx, var_name in enumerate(variables):
        # Create a new figure
        fig, axes = plt.subplots(1, len(set(g_index)), figsize=(12, 6), sharey='row')
        
        # Loop over each cluster
        for cluster in list(set(g_index)):
            if len(set(g_index)) > 1:
                ax = axes[cluster]
            else:
                ax = axes
            
            subjects_in_cluster = np.where(g_index == cluster)[0]
            
            # Plot trajectories
            for subject in subjects_in_cluster:
                trajectory = States_original[subject, :, var_idx]
                ax.plot(trajectory, alpha=0.1, color='black')
            
            # Plot change point
            change_point = changepoints[subject]
            ax.axvline(change_point, color='red', linestyle='--', linewidth=2)
            
            # Plot means before and after change point
            mean_before = np.mean(States_original[subjects_in_cluster, :change_point, var_idx])
            mean_after = np.mean(States_original[subjects_in_cluster, change_point:, var_idx])
            
            ax.hlines(mean_before, xmin=0, xmax=change_point, color='green', linestyle='--', linewidth=2)
            ax.hlines(mean_after, xmin=change_point, xmax=States.shape[1], color='yellow', linestyle='--', linewidth=2)
    
            # ax.set_title(f'Cluster {cluster} - {var_name}')
            ax.set_title(f'Cluster {cluster+1}')
            ax.set_xlabel('Time Step')
            if cluster == 0:
                ax.set_ylabel(y_label_name[var_name])
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{method}_{var_name}_cluster_changepoints_originalscale.png")
        plt.close(fig)  # 
    #%%
    confusion_matrix = pd.crosstab(specialty, g_index, rownames=['Specialty'], colnames=['Cluster'])
    print(confusion_matrix)
    if training_phrase == "plot":
        exit()
    
    #%% ope
    os.chdir(ope_path)
    if not refit:
        sys.stdout = open("log.txt", "a")
    else:
        sys.stdout = open("log.txt", "w")
    np.random.seed(seed+1)
    random.seed(seed+1)

    ope_value, df_cluster_rewards = evaluate(States, Actions, Rewards, 
                                             changepoints, g_index)
    

    sys.stdout.flush()
    #%% save data
    with open(file_name, 'wb') as f:
        pickle.dump({"ope_individual": df_cluster_rewards,
                     "ope_value":ope_value
                     }, f)
    
    sys.stdout.close()
    sys.stdout=stdoutOrigin

#%%
if __name__ == '__main__':
    main()
