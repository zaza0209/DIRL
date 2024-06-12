#!/usr/bin/python
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

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
    elif Kl_fun == "sqrt(N)":
        Kl = np.sqrt(N)*K

    # print("kl", Kl)
    _, indicesList, occurCount = np.unique(g_index, return_index=True, return_counts=True)
    # print('c',occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * C * np.log(np.sum(T-1 -changepoints)))

    print('loss =', loss)
    ic = loss - Kl
    print("Kl =", Kl)
    print("IC =", ic)
    return ic
#%%

# number of individuals
N = 477
# number of time points (starting from 0, to 26 weeks)
T = 27
# the candidate values of K
Ks = [1,2,3,4,5,6,7,8]
file_name = "D:/OneDrive/PhD/DIRL/IHS/simu/simu_anonymous/ic/result_with_llk.dat"
out = pickle.load(open(file_name, "rb"))

# the coefficient before sqrt(|C_k| / \tau_k)
q = 10
dat = pd.DataFrame(columns=['K', 'IC'])
row_idx = 0
Kl_fun='sqrt(N)'
for K in Ks:
    # obtain change points under K clusters
    changepoints = out['changepoints_all'][K]
    # obtain cluster indices under K clusters
    clusters = out['clusters_all'][K]
    # obtain unnormalized loglikelihood
    loss_unnormalized = out['loss_unnormalized_all'][K]
    # obtain normalized loglikelihood (unnomalized llk divided by tau_k)
    loss_normalized = out['loss_normalized_all'][K]
    # obtain the dimension of nonzero coefficients in each cluster
    dfs = out['df_all'][K]
    # obtain the residual variances
    # sigma = out['sigma_all'][K-1]
    
    '''
    # obtain the observed outcome in the regression model for change point detection
    # i.e., S_{i,t,j} the weekly average variable j of individual i in week t.
    # j = 1 for step counts at week t, j = 2 for sleep at week t, j = 3 for mood at week t,
    # j = 4 for step counts at week t - 1
    y_observed = out['y_observed_all'][K]
    # y_observed is a dictionary with the values of K as the keys
    # e.g., y_observed = out['y_observed_all'][3] contains observed outcome under K = 3, where
    # the k-th element y_observed[k] is a |C_k| by tau_k by p array, k = 1, ..., K. Here, |C_k|
    # is the size of cluster k, tau_k is the number of time points after the change point, and
    # p = 4 is the dimension of the state

    # obtain the predicted outcome in the regression model for change point detection
    # i.e., S_{i,t,1} the weekly average step counts of individual i in week t.
    # y_predicted has the same structure as y_observed
    y_predicted = out['y_predicted_all'][K]

    # the regression model from which we obtain the predictions
    # model = out['model_all'][K] contains a list of linear regression with SCAD penalty under K
    # clusters. len(model) = K, where model[k] is the fitted model for each cluster k = 1, ..., K
    model = out['model_all'][K]

    # Now calculate each individual's average least square loss over time after the change point
    least_square_loss_K = []
    for k in range(K):
        least_square_loss_K.append(np.mean((y_observed[k] - y_predicted[k]) ** 2, axis = 1))
    # concatenating all individuals into a 477 by 4 array. The last column should be close to 0
    least_square_loss_K = np.vstack(least_square_loss_K)
    '''
    value = IC(loss_normalized, changepoints, clusters, N, T, K, dfs, q, C=2, Kl_fun=Kl_fun)
    dat.loc[row_idx] = [K, value]
    row_idx += 1


#%% create a line plot of IC
x = np.array(Ks)
# indices = np.array([0,2,3,4,5,6]) #np.arange(0, len(K_list) * 3, step = 3)
y = dat['IC'].to_list() #[indices]
fig, ax = plt.subplots()
ax.plot(x, y, '-')
ax.xaxis.set_ticks(x)
ax.yaxis.set_ticks(y)
ax.set_xlabel('Number of clusters K')
ax.set_ylabel('Value')
# ax.set_title('C = ' + str(C))
# ax.set_title('Normalized loglikelihood (divided by tau_k)')
# ax.set_title('Unnormalized loglikelihood (Not divided by tau_k)')
ax.set_title(Kl_fun+'criterion with normalized loglikelihood' + ' (q = ' + str(q) + ')')
# add IC values next to points
for i, txt in enumerate(y):
    ax.annotate(round(txt, 2), (x[i], y[i]))
plt.show()


#%% print variances
for K in Ks:
    sigma = out['sigma_all'][K]
    sigma_to_print = np.round(sigma, 2)
    sigma_to_print = sigma_to_print.astype('str')
    sigma_to_print = ", ".join(sigma_to_print.tolist())
    print("K = " + str(K) + ": sigma^2_k = ", sigma_to_print)

#%% print likelihoods
for K in Ks:
    loss_normalized = out['loss_normalized_all'][K]
    loss_to_print = np.round(loss_normalized, 2)
    loss_to_print = loss_to_print.astype('str')
    loss_to_print = ", ".join(loss_to_print.tolist())
    print("K = " + str(K) + ": loss_k = ", loss_to_print)





