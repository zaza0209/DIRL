# import sys, re
from joblib import Parallel, delayed
import numpy as np
from collections import namedtuple
import os, pickle

# %% IC
def h_in_IC(changepoints, T, h='1'):
    '''
    The bonus term for changepoints location in IC
    '''
    if h == '1':
        return (np.sum(T - 1 - changepoints) / np.log(np.sum(T - 1 - changepoints)))
    elif h == 'sqrt':
        return (np.sqrt(np.sum(T - 1 - changepoints) / np.log(np.sum(T - 1 - changepoints))))


def IC(loss, changepoints, g_index, N, T, K, C=1, Kl_fun="sqrtN",
       h='1', C_K = 0, ic_T_dynamic=0):
    '''
    Parameters
    ----------
    Kl_fun : Penalty on K. The default is 'sqrtN' if IC is calculated at the end of the algorithm.
        If tuning K at each iteration, Kl_fun should be 'Nlog(NT)/T'.
    h : TYPE, optional
        DESCRIPTION. The default is '1'.
    C_K: The coefficient in h function. Since we no long have the h term in the IC, the default is 0.
    C : The coefficient in the penalty function Kl_fun. The default is 2. Note that I'm not sure how to set the default value. 
          In theory, this value should be of order O(log(NT)). I've tried on my side C_K=2 works well in an example where the penalty function with C_K=2 happends to equal to that in the BIC.
          If C=2 doesn't work for you, you may consider change it to something else like O(log(NT)), O(sqrt(log(NT))), O(NT).
          
    '''
    # K = len(set(g_index))
    print("Kl_fun = ", Kl_fun)
    print('K', K, 'N', N)
    T_mean = np.mean(T - 1 - changepoints)
    if ic_T_dynamic:
        T = T_mean
        
    if Kl_fun == 'log':
        Kl = K * np.log(np.sum(T - 1 - changepoints))
    elif Kl_fun == "sqrt":
        Kl = K * np.sqrt(np.sum(T - 1 - changepoints))
    elif Kl_fun == 'logN':
        Kl = K * np.log(N)
    elif Kl_fun == 'Nlog(NT)/T':
        Kl = C* K* N * np.log(N*T)/T
        print('T', T, "N * np.log(N*T)/T", N * np.log(N*T)/T)
    elif Kl_fun == "sqrt{Nlog(NT)/T}":
        Kl = C*K*np.sqrt(N*np.log(N*T)/T)
        print('T', T,'np.sqrt(N*np.log(N*T)/T)',np.sqrt(N*np.log(N*T)/T))
    elif Kl_fun =="sqrtN":
        Kl = C* K* np.sqrt(N)
    elif Kl_fun == "BIC":
        loss = N* np.log(loss/N)
        Kl = K * np.log(N)
    # print("kl", Kl)
    _, indicesList, occurCount = np.unique(g_index, return_index=True, return_counts=True)
    # print('c',occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * C * np.log(np.sum(T-1 -changepoints)))
    # if loss_fun == 'no_scale':
    #     loss = loss / np.mean(T - 1 - changepoints)
    ic=loss - Kl #+ occurCount.dot((T - 1 - changepoints)[np.s_[indicesList]]) / (N * T) * C_K * h_in_IC(changepoints,
                                                                                                          # T, h=h)

    # print('(T-1 -changepoints)[np.s_[indicesList]]',(T-1 -changepoints)[np.s_[indicesList]])
    # print('np.log(np.sum(T-1 -changepoints)',np.log(np.sum(T-1 -changepoints)))
    print('loss', loss, "ic", ic, ', ', K, '*l', Kl)
    return ic

def paramInIC(model, N, K, T, include_path_loss=0):
    if not include_path_loss:
        Kl_mat = K * np.log(np.sum(T - 1 - model['changepoint']))
        Ck, indicesList, occurCount = np.unique(model['group'], return_index=True, return_counts=True)
        c_mat = occurCount.dot((T - 1 - model['changepoint'])[np.s_[indicesList]]) / (N * T) * h_in_IC(
            model['changepoint'], T)
        print("occurCount.dot((T-1 -model['changepoint'])[np.s_[indicesList]])/(N*T)",
              occurCount.dot((T - 1 - model['changepoint'])[np.s_[indicesList]]) / (N * T),
              "h_in_IC(model['changepoint'], T)", h_in_IC(model['changepoint'], T))
        print(c_mat)
        return Kl_mat, c_mat
    else:
        Kl_mat_list = []
        c_mat_list = []
        cp_eachiter_rev = model['changepoint_eachiter'][:, range(model['changepoint_eachiter'].shape[1] - 1, -1, -1)]
        g_eachiter_rev = model['g_index_eachiter'][:, range(model['g_index_eachiter'].shape[1] - 1, -1, -1)]
        for ii in range(min(model['changepoint_eachiter'].shape[1], model['g_index_eachiter'].shape[1])):
            Kl_mat_list.append(K * np.log(np.sum(T - 1 - cp_eachiter_rev[:, ii])))
            Ck, indicesList, occurCount = np.unique(g_eachiter_rev[:, ii], return_index=True, return_counts=True)
            c_mat_list.append(occurCount.dot((T - 1 - cp_eachiter_rev[:, ii])[np.s_[indicesList]]) / (N * T) * h_in_IC(
                cp_eachiter_rev[:, ii], T))
        return Kl_mat_list[::-1], c_mat_list[::-1]


# %% threshold in changepoint deteciton estimation
def estimate_threshold(N, kappa, df, nthread=5, B=5000, alpha=0.01,
                       save_path=None):
    sample_stat=None
    if save_path is not None:
        data_file_name = save_path +'/sample_stat_N'+str(N)+'_kappa'+str(kappa)+'_B'+str(B)+'.dat'
        if os.path.exists(data_file_name):
            with open(data_file_name, "rb") as f:
                sample_stat = pickle.load(f)
            
            
    def sample_by_group(condition, nB_per_group):
        random_state = condition[1]
        rng = np.random.RandomState(random_state)
        X = rng.multivariate_normal(mean, cov, size=[N, kappa, nB_per_group])
        stat_max_by_u = 0.0
        for u in range(kappa - 1, 0, -1):
            mu1 = np.sum(X[:, :u, :], axis=(0, 1))
            mu2 = np.sum(X[:, u:, :], axis=(0, 1))
            cusum1 = np.sqrt((kappa - u)/(N*u*kappa))*mu1 
            cusum2 = np.sqrt(u/(N*(kappa-u)*kappa))*mu2
            stat = np.linalg.norm(cusum1 -cusum2, ord=2, axis=-1)**2
            stat_max_by_u = np.maximum(stat, stat_max_by_u)
        return stat_max_by_u


    mean = np.zeros(df)
    cov = np.eye(df)
    # max number of points per group
    n_points = int(500000)
    
    if N * kappa * B <= n_points and sample_stat is None:
        random_states = np.random.randint(np.iinfo(np.int32).max, size=B)
        conditions = list(zip(range(B), random_states))
        sample_stat = Parallel(n_jobs=nthread, prefer="threads")(delayed(sample_by_group)(condition, 1) 
                                                                 for condition in conditions)
        if save_path is not None:
            with open(data_file_name, "wb") as f:
                pickle.dump(sample_stat, f)
    elif sample_stat is None:
        # number of groups of points
        n_groups = int(np.floor(N * kappa * B / n_points))
        # for the remaining number of points
        # number of samples per group
        nB_per_group = max(10, int(np.floor(B / n_groups)))
        nB_remaining = B - nB_per_group * n_groups

        def f(x):
            return np.linalg.norm(x, ord=2) ** 2


        random_states = np.random.randint(np.iinfo(np.int32).max, size=n_groups+1)
        conditions = list(zip(range(n_groups+1), random_states))
        sample_stat = Parallel(n_jobs=nthread, prefer="threads")(delayed(sample_by_group)(condition, nB_per_group) 
                                                                 for condition in conditions[:-1])
        sample_stat = np.array(sample_stat).flatten()
        sample_stat2 = sample_by_group(conditions[-1], nB_remaining)
        sample_stat = np.concatenate([sample_stat, sample_stat2])
        
        if save_path is not None:
            with open(data_file_name, "wb") as f:
                pickle.dump(sample_stat, f)
    
    threshold = np.percentile(sample_stat, (1 - alpha) * 100)
    
        
    return threshold, sample_stat

def chi2_threshold(N, kappa, df, nthread=5, B=5000, alpha=0.01):
    def run_one():
        chi2_variables = np.random.chisquare(df=df, size=kappa)
        return np.max(chi2_variables)
    sample_stat = Parallel(n_jobs=nthread)(delayed(run_one)() for i in range(B))
    threshold = np.percentile(sample_stat, (1 - alpha) * 100)
    return threshold, sample_stat

# %% time series clustering
# from dtaidistance import dtw
# from scipy.cluster.hierarchy import single, complete, average, ward, fcluster
# import pandas as pd


# # from scipy.cluster.hierarchy import fcluster
# # import pandas as pd
# def my_hierachy(States, K, distance_metric='correlation', linkage="average"):
#     if distance_metric == "correlation":
#         # Reshape the data so each series is a column and call the dataframe.corr() function
#         # distance_matrix = pd.concat([series for series in X['dim_0'].values], axis=1).corr()
#         distance_matrix = pd.DataFrame(np.vstack(States[:, :, :].reshape(States.shape[0], 1, -1)).T).corr()
#     elif distance_metric == "DWT":
#         # Italy Power Demand time series are loaded in a pd.Series format.
#         # The dtw_distance function expects series to be shaped as a (l, m) array,
#         # where l=length of series, m=# dimensions
#         # series_list = X['dim_0'].values
#         series_list = [None] * States.shape[0]
#         for i in range(len(series_list)):
#             # length = len(series_list[i])
#             series_list[i] = States[i, :, :].flatten().reshape(-1, 1)

#         # Initialize distance matrix
#         n_series = len(series_list)
#         distance_matrix = np.zeros(shape=(n_series, n_series))

#         # Build distance matrix
#         for i in range(n_series):
#             for j in range(n_series):
#                 x = series_list[i]
#                 y = series_list[j]
#                 if i != j:
#                     dist = dtw.distance_fast(x.flatten(), y.flatten())
#                     distance_matrix[i, j] = dist

#     def hierarchical_clustering(dist_mat, method='complete'):
#         if method == 'complete':
#             Z = complete(distance_matrix)
#         if method == 'single':
#             Z = single(distance_matrix)
#         if method == 'average':
#             Z = average(distance_matrix)
#         if method == 'ward':
#             Z = ward(distance_matrix)
#         # fig = plt.figure(figsize=(16, 8))
#         # dn = dendrogram(Z)
#         # plt.title(f"Dendrogram for {method}-linkage with correlation distance")
#         # plt.show()
#         return Z

#     linkage_matrix = hierarchical_clustering(distance_matrix, method=linkage)

#     # select maximum number of clusters
#     cluster_labels = fcluster(linkage_matrix, K, criterion='maxclust')
#     # print(np.unique(cluster_labels))
#     # #>> 4 unique clusters
#     # cluster_labels = fcluster(linkage_matrix, 10, criterion='maxclust')
#     # print(np.unique(cluster_labels))
#     # >> 10 unique clusters

#     # hand-select an appropriate cut-off on the dendrogram
#     # cluster_labels = fcluster(linkage_matrix, 600, criterion='distance')
#     # print(np.unique(cluster_labels))
#     # #>> 3 unique clusters
#     # cluster_labels = fcluster(linkage_matrix, 800, criterion='distance')
#     # print(np.unique(cluster_labels))
#     # >> 2 unique clusters
#     cluster_labels = (cluster_labels - 1).astype(int)
#     return cluster_labels







