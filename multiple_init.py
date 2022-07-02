'''
multiple initialization methods
'''
import sys, os, pickle
import numpy as np
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/")
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
import simu.simu_mean_detect as mean_detect

#%%
def setpath(seed, method):
    os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
    append_name = '_N' + str(N) + '_1d'
    if not os.path.exists('results/value_evaluation'):
        os.makedirs('results/value_evaluation', exist_ok=True)
    data_path = 'results/value_evaluation/coef'+re.sub("\\ ", "",re.sub("\\.", "", re.sub("\\]","", re.sub("\\[", "", re.sub("\\, ", "", str(coef1))))))+'/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) + \
                                                  append_name
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    data_path += '/sim_result' + method + '_gamma' + re.sub("\\.", "", str(gamma)) + \
                  append_name + '_' + str(seed)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)
    return
def save_data(file_name, tmp,seed, method):
    setpath(seed, method)
    # print('save data')
    with open(file_name, "wb") as f:
        pickle.dump({'iter_num' : tmp.iter_num, 'group':tmp.g_index,
                     'changepoint' :tmp.changepoints, 
                     'changepoint_eachiter':tmp.changepoint_eachiter,
                     'g_index_eachiter':tmp.g_index_eachiter,
                     'loss':tmp.loss,
                     "ic":tmp.IC}, f)
    return
#%%
def random_changepoints(States, Actions, K, seed, cp_num=5,nthread=3, C=1):
    '''
    Parameters
    ----------
    cp_num : TYPE, optional
        the number of random initial change points. The default is 5.

    '''
    T = States.shape[1]
    N = States.shape[0]
    IC_max = -1000
    for i in range(cp_num):
        changepoints_init = np.random.choice(range(T-1), size = 1) * np.ones(N)
        changepoints_init = changepoints_init.astype(int)
        out = mean_detect.fit(States, Actions, K=K, example="cdist", 
                              changepoints_init=changepoints_init, clustering_warm_start=1,
                              seed = seed, nthread=nthread, max_iter=5)
        IC_tmp = out.IC
        if i == 0 or IC_max < IC_tmp:
            IC_max = IC_tmp
            best_res = out
        file_name = 'random_cp'+ str(i) + '.dat'
        # save_data(file_name, out, seed, 'random_cp')
    return best_res, IC_max

def varying_K(States, Actions, K, seed, init_cluster_method='gmr', distance_metric = 'correlation', linkage ='average', nthread=3):
    out = mean_detect.fit(States, Actions, example="cdist", init="clustering", K=K,
                                seed = seed, nthread=nthread, max_iter=5,
                                init_cluster_range=10,
                                changepoints_init = None, 
                                clustering_warm_start=1, init_cluster_method=init_cluster_method, 
                                distance_metric=distance_metric, linkage=linkage)
    file_name = 'init_clustering_method'+init_cluster_method+'_K'+ str(K) + '.dat'
    # save_data(file_name, out, seed, 'random_cp')
    return out, out.IC

#%%
def run_init(States, Actions, seed, method_list = ['random_cp', 'gmr','kmeans', 'hierarchy'], K_list = range(2,6), cp_num=5, distance_metric = 'correlation', linkage ='average', nthread=3):
    IC_max = None
    for K in K_list:
        if 'random_cp' in method_list:
            res, IC = random_changepoints(States, Actions, K, seed,cp_num,nthread)
            if IC_max is None or IC_max < IC:
                IC_max = IC
                best_res = res
        if 'gmr' in method_list:
            res, IC=varying_K(States, Actions, K, seed, 'gmr', nthread = nthread)
            if IC_max is None or IC_max < IC:
                IC_max = IC
                best_res = res
        if 'kmeans' in method_list:
            if distance_metric == 'correlation':
                distance_metric = 'euclide'
            res, IC=varying_K(States, Actions, K, seed, 'kmeans',distance_metric, nthread = nthread)
            if IC_max is None or IC_max < IC:
                IC_max = IC
                best_res = res
        if 'hierarchy' in method_list:
            res, IC=varying_K(States, Actions, K, seed, 'hierarchy', distance_metric, linkage,nthread = nthread)
            if IC_max is None or IC_max < IC:
                IC_max = IC
                best_res = res
    return best_res
    