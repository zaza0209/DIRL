# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

# %%
def IC(loss_by_K, changepoints, g_index, N, T, K, dfs, q = 1.96, C=0, Kl_fun='log(NT)/T'):

    loss = 0
    for k in range(K):
        loss += loss_by_K[k]

    if Kl_fun == 'log(NT)/T':
        Kl = C * K * np.log(N*T)/T
    elif Kl_fun == 'chi2':
        # dat_out = pd.DataFrame({'cluster': g_index, 'changepoint': changepoints})
        # print('g_index =', g_index)
        cluster_cp = pd.crosstab(g_index, changepoints)
        cols = cluster_cp.columns
        bt = cluster_cp.apply(lambda x: x > 0)
        cluster_cp = np.concatenate(bt.apply(lambda x: list(cols[x.values]), axis=1))
        _, counts = np.unique(g_index, return_counts=True)

        n_timepoint_after_cp = T - 1 - cluster_cp
        # dim_covariate = 24
        Kl = 0
        for k in range(K):
            dim_covariate = dfs[k]
            chi2_q = chi2.ppf(0.9, dim_covariate)  # dfs[k]
            sigma_k = -loss_by_K[k] / (n_timepoint_after_cp[k] * counts[k] - dim_covariate)
            print("sigma_k =", sigma_k)
            print("dim_covariate =", dim_covariate)
            Kl += sigma_k * chi2_q / n_timepoint_after_cp[k] + 2*q * np.sqrt(counts[k] / n_timepoint_after_cp[k])
    elif Kl_fun == '0':
        Kl = 0

    # print("kl", Kl)
    _, indicesList, occurCount = np.unique(g_index, return_index=True, return_counts=True)
    # print('c',occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * C * np.log(np.sum(T-1 -changepoints)))

    print('loss =', loss)
    ic = loss + Kl
    print("Kl =", Kl)
    print("IC =", ic)
    return ic

# %%
# number of individuals
N = 477
# number of time points (starting from 0, to 26 weeks)
T = 27
# the candidate values of K
Ks = [1,2,3,4,5,6,7,8]
file_name = "D:/OneDrive/PhD/DIRL/IHS/simu/simu_anonymous/ic/result_with_llk.dat"
out = pickle.load(open(file_name, "rb"))

# %%
# obtain change points under K clusters
changepoints = out['changepoints_all']
# obtain cluster indices under K clusters
clusters = out['clusters_all']
# obtain unnormalized loglikelihood
loss_unnormalized = out['loss_unnormalized_all']
# obtain normalized loglikelihood (unnomalized llk divided by tau_k)
loss_normalized = out['loss_normalized_all']
# obtain the dimension of nonzero coefficients in each cluster
dfs = out['df_all']
# obtain the residual variances
sigma = out['sigma_all']

# %%
loss_n = np.zeros(8)
loss_un = np.zeros(8)
cluster_cp = [None]*8
cluster_no = [None]*8
sigma2 = [None]*8
for K in Ks:
    loss_n[K-1] += sum(loss_normalized[K])
    loss_un[K-1] += sum(loss_unnormalized[K])
    cp = pd.crosstab(clusters[K], changepoints[K])
    cols = cp.columns
    bt = cp.apply(lambda x: x > 0)
    cluster_cp[K-1] = T-1-np.concatenate(bt.apply(lambda x: list(cols[x.values]), axis=1))
    _, counts = np.unique(clusters[K], return_counts=True)
    cluster_no[K-1] = counts
    sigma2[K-1] = -loss_unnormalized[K]/(cluster_cp[K-1]*cluster_no[K-1]-dfs[K])

# %%
loss_n

# %%
sigma2

# %%
cluster_cp

# %%
cluster_no

# %%
penalty_n = [None]*8
penalty_ns = np.zeros(8)
for K in Ks:
    penalty_n[K-1] = sigma2[K-1] * np.sqrt(cluster_no[K-1]/cluster_cp[K-1])
    penalty_ns[K-1] = np.sum(penalty_n[K-1])

# %%
penalty_ns

# %%
loss_n - penalty_ns*2*1.96

# %%
cluster_ns = np.zeros(8)
for K in Ks: 
    cluster_ns[K-1] = np.sum(np.sqrt(cluster_no[K-1]))

# %%
cluster_ns

# %%
loss_n = np.array([-910.2592503162963, -839.1226686380818,-804.6847070892476,
          -781.6860318450282, -757.0795084643895, -743.7454747676743, 
          -726.8653177935311])
loss_n - np.array(range(1, 8))*N*np.sqrt(np.log(N*T)/T)*0.1

## ?
# loss_n - np.array(range(1, K+1))*np.sqrt(np.log(N*T)*N)
# %%
clusters

# %%
len(clusters[1])

# %%



