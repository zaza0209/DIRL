# import sys, re
from joblib import Parallel, delayed
import numpy as np


# %% IC
def h_in_IC(changepoints, T, h='1'):
    '''
    The bonus term for changepoints location in IC
    '''
    if h == '1':
        return (np.sum(T - 1 - changepoints) / np.log(np.sum(T - 1 - changepoints)))
    elif h == 'sqrt':
        return (np.sqrt(np.sum(T - 1 - changepoints) / np.log(np.sum(T - 1 - changepoints))))


def IC(loss, changepoints, g_index, N, T, K, C=0, loss_fun='no_scale', Kl_fun='Nlog(NT)/T', h='1', C_K = 2):
    '''
    Parameters
    ----------
    Kl_fun : Penalty on K. The default is 'NlogNT/T'. In older version of IC, this is 'logN'
    h : TYPE, optional
        DESCRIPTION. The default is '1'.
    C: The coefficient in h function. Since we no long have the h term in the IC, the default is 0.
    C_K : The coefficient in the penalty function Kl_fun. The default is 2. Note that I'm not sure how to set the default value. 
          In theory, this value should be of order O(sqrt(log(NT))). I've tried on my side C_K=2 works well in an example where the penalty function with C_K=2 happends to equal to that in the BIC.
          If C_K=2 doesn't work for you, you may consider change it to something else like O(sqrt(log(NT))).

    '''
    # K = len(set(g_index))
    # print("Kl_fun = ", Kl_fun)
    if Kl_fun == 'log':
        Kl = K * np.log(np.sum(T - 1 - changepoints))
    elif Kl_fun == "sqrt":
        Kl = K * np.sqrt(np.sum(T - 1 - changepoints))
    elif Kl_fun == 'logN':
        Kl = K * np.log(N)
    elif Kl_fun == 'Nlog(NT)/T':
        Kl = C* K* N * np.log(N*T)/T
    # print("kl", Kl)
    _, indicesList, occurCount = np.unique(g_index, return_index=True, return_counts=True)
    # print('c',occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * C * np.log(np.sum(T-1 -changepoints)))
    if loss_fun == 'no_scale':
        loss = loss / np.mean(T - 1 - changepoints)
    print('loss', loss)
    # print('(T-1 -changepoints)[np.s_[indicesList]]',(T-1 -changepoints)[np.s_[indicesList]])
    # print('np.log(np.sum(T-1 -changepoints)',np.log(np.sum(T-1 -changepoints)))
    print("ic",
          loss - Kl + occurCount.dot((T - 1 - changepoints)[np.s_[indicesList]]) / (N * T) * C * h_in_IC(changepoints,
                                                                                                         T), ', ', K,
          '*l', Kl, ', ', C, '* h=',
          C * occurCount.dot((T - 1 - changepoints)[np.s_[indicesList]]) / (N * T) * h_in_IC(changepoints, T))
    return loss - Kl + occurCount.dot((T - 1 - changepoints)[np.s_[indicesList]]) / (N * T) * C * h_in_IC(changepoints,
                                                                                                          T, h=h)


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
def estimate_threshold(N, kappa, df, nthread=5, B=5000, alpha=0.01, seed=5):
    def run_one_normal(X, u):
        # np.random.seed(seed)
        mu1 = np.mean(X[:, :u, :], axis=(0, 1))
        mu2 = np.mean(X[:, u:, :], axis=(0, 1))
        stat = u * (kappa - u) / (kappa ** 2) * np.linalg.norm(mu1 - mu2, ord=2) ** 2
        return stat

    mean = np.zeros(df)
    cov = np.eye(df)
    if N * kappa * B < 5000:
        X_list = np.random.multivariate_normal(mean, cov, size=[N, kappa, B])
        sample_stat = Parallel(n_jobs=nthread)(delayed(run_one_normal)(X_list[:, :, i, :], u) for i in range(B)
                                               for u in range(kappa - 1, 0, -1))
    else:
        # sample_stat = Parallel(n_jobs=nthread, backend ='threading')(delayed(run_one_normal)(np.random.multivariate_normal(mean, cov, size = [N,kappa,1]), u) for i in range(B)
        #                                         for u in range(kappa-1, 0, -1))

        # max number of points per group
        n_points = int(500000)
        # number of groups of points
        n_groups = int(np.floor(N * kappa * B / n_points))
        # for the remaining number of points
        n_remaining_points = int(N * kappa * B - n_groups * n_points)
        # number of samples per group
        nB_per_group = max(10, int(np.floor(B / n_groups)))
        nB_remaining = B - nB_per_group * n_groups

        def f(x):
            return np.linalg.norm(x, ord=2) ** 2

        def sample_by_group(condition, nB_per_group):
            random_state = condition[1]
            rng = np.random.RandomState(random_state)
            X = rng.multivariate_normal(mean, cov, size=[N, kappa, nB_per_group])
            # X = np.random.multivariate_normal(mean, cov, size=[N, kappa, nB_per_group])
            stat_max_by_u = 0.0
            for u in range(kappa - 1, 0, -1):
                mu1 = np.mean(X[:, :u, :], axis=(0, 1))
                mu2 = np.mean(X[:, u:, :], axis=(0, 1))
                stat = u * (kappa - u) / (kappa ** 2) * np.array(list(map(f, mu1 - mu2)))
                stat_max_by_u = np.maximum(stat, stat_max_by_u)
            return stat_max_by_u

        # generate random states
        np.random.seed(seed*kappa)
        random_states = np.random.randint(np.iinfo(np.int32).max, size=n_groups+1)
        conditions = list(zip(range(n_groups+1), random_states))
        sample_stat = Parallel(n_jobs=nthread, prefer="threads")(delayed(sample_by_group)(condition, nB_per_group) for condition in conditions[:-1])
        # print("sample_stat =", sample_stat)
        sample_stat = np.array(sample_stat).flatten()
        sample_stat2 = sample_by_group(conditions[-1], nB_remaining)
        sample_stat = np.concatenate([sample_stat, sample_stat2])

    # sample_stat = np.max(np.array(sample_stat).reshape([B, -1]), axis = 1)
    threshold = np.percentile(sample_stat, (1 - alpha) * 100)
    return threshold, sample_stat


# %% time series clustering
from dtaidistance import dtw
from scipy.cluster.hierarchy import single, complete, average, ward, fcluster
import pandas as pd


# from scipy.cluster.hierarchy import fcluster
# import pandas as pd
def my_hierachy(States, K, distance_metric='correlation', linkage="average"):
    if distance_metric == "correlation":
        # Reshape the data so each series is a column and call the dataframe.corr() function
        # distance_matrix = pd.concat([series for series in X['dim_0'].values], axis=1).corr()
        distance_matrix = pd.DataFrame(np.vstack(States[:, :, :].reshape(States.shape[0], 1, -1)).T).corr()
    elif distance_metric == "DWT":
        # Italy Power Demand time series are loaded in a pd.Series format.
        # The dtw_distance function expects series to be shaped as a (l, m) array,
        # where l=length of series, m=# dimensions
        # series_list = X['dim_0'].values
        series_list = [None] * States.shape[0]
        for i in range(len(series_list)):
            # length = len(series_list[i])
            series_list[i] = States[i, :, :].flatten().reshape(-1, 1)

        # Initialize distance matrix
        n_series = len(series_list)
        distance_matrix = np.zeros(shape=(n_series, n_series))

        # Build distance matrix
        for i in range(n_series):
            for j in range(n_series):
                x = series_list[i]
                y = series_list[j]
                if i != j:
                    dist = dtw.distance_fast(x.flatten(), y.flatten())
                    distance_matrix[i, j] = dist

    def hierarchical_clustering(dist_mat, method='complete'):
        if method == 'complete':
            Z = complete(distance_matrix)
        if method == 'single':
            Z = single(distance_matrix)
        if method == 'average':
            Z = average(distance_matrix)
        if method == 'ward':
            Z = ward(distance_matrix)
        # fig = plt.figure(figsize=(16, 8))
        # dn = dendrogram(Z)
        # plt.title(f"Dendrogram for {method}-linkage with correlation distance")
        # plt.show()
        return Z

    linkage_matrix = hierarchical_clustering(distance_matrix, method=linkage)

    # select maximum number of clusters
    cluster_labels = fcluster(linkage_matrix, K, criterion='maxclust')
    # print(np.unique(cluster_labels))
    # #>> 4 unique clusters
    # cluster_labels = fcluster(linkage_matrix, 10, criterion='maxclust')
    # print(np.unique(cluster_labels))
    # >> 10 unique clusters

    # hand-select an appropriate cut-off on the dendrogram
    # cluster_labels = fcluster(linkage_matrix, 600, criterion='distance')
    # print(np.unique(cluster_labels))
    # #>> 3 unique clusters
    # cluster_labels = fcluster(linkage_matrix, 800, criterion='distance')
    # print(np.unique(cluster_labels))
    # >> 2 unique clusters
    cluster_labels = (cluster_labels - 1).astype(int)
    return cluster_labels
