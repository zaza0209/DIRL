"""
Detect change points and clusters in mean.
"""
from sklearn.cluster import KMeans
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from collections import namedtuple
from scipy.linalg import block_diag
from scipy.stats import norm
from joblib import Parallel, delayed
import functions.utilities as ut
import random
from datetime import datetime
import time
from scipy import sparse
# from skglm import GeneralizedLinearEstimator
# from skglm.penalties import MCPenalty, SCAD
# from skglm.datafits import Quadratic
# from itertools import product
from copy import copy
# import pandas as pd

#%%
# def goodnessofChangepoint(States, Actions, changepoints, g_index, cv, param_grid, nfold, nthread, penalty_function, seed):
#     '''
#     Calculate loglikelihood of the result if the algorithm ends in change point detection
#     :param States:
#     :param Actions:
#     :param changepoints:
#     :param g_index:
#     :return:
#     '''
#     p = States.shape[2]
#     T = States.shape[1]
#     N = States.shape[0]
#     g_index = g_index.astype(int)
#     K = len(set(g_index))
#     changepoints = changepoints.astype(int)
#     cluster_cp = pd.crosstab(g_index, changepoints)
#     cols = cluster_cp.columns
#     bt = cluster_cp.apply(lambda x: x > 0)
#     cluster_cp = np.concatenate(bt.apply(lambda x: list(cols[x.values]), axis=1))
#     # np.unique(changepoints, return_counts = True)

#     #%% compute test statistic for each cluster k, taking maximum over u
#     def run_k(k, nthread):
#         tau_k = cluster_cp[k]
#         States_k = States[(g_index == k), tau_k:, :]
#         Actions_k = Actions[(g_index == k), tau_k:]
#         dfs = []
#         # select tuning parameters
#         if States_k.shape[0] >= 10 and cv:
#             cv_result = grid_search_cv(States_k, Actions_k, param_grid, nfold, nthread, penalty_function, seed)
#             param = cv_result['selected_param']
#         else:
#             param = {'alpha': max(param_grid['alpha']), 'gamma': max(param_grid['gamma'])}

#         #%% fit model on all data from T-kappa to T
#         States_current = States_k[:, :-1, :]
#         N_k = States_current.shape[0]
#         States_next = States_k[:, 1:, :]
#         Actions_current = Actions_k
#         X = []
#         y = []
#         for i in range(States_current.shape[0]):
#             poly = PolynomialFeatures(2, interaction_only=True)
#             mat_tmp = poly.fit_transform(np.vstack([Actions_current[i, :], States_current[i, :, range(p)]]).T)
#             mat = np.delete(mat_tmp, np.s_[2 + p * 2:mat_tmp.shape[1]], 1)
#             X.append(mat)
#             y.append(States_next[i, :, :]) # y[0].shape = (T - tau_k, p)
#         # perform regression
#         # len(X) = N_k, where X[0].shape = (T - tau_k, 2p + 2)
#         # note that np.array(X).shape = (N_k, T - tau_k, 2p + 2)
#         X = np.vstack(np.array(X)) # now X.shape = (N_k * (T - tau_k), 2p + 2)
#         X = np.kron(np.eye(p, dtype=int), X) # X.shape = (N_k * (T - tau_k) * p, (2p + 2) * p)
#         X = sparse.csr_matrix(X) # X.shape = (N_k * (T - tau_k) * p, (2p + 2) * p)
#         # np.array(y).shape = (N_k, T - tau_k, p)
#         y = np.vstack(np.array(y)).T.reshape(-1, )
#         # For S_{i,t,j}, Y = (S_{1,2,1}, S_{1,3,1}, ..., S_{1,T,1}, S_{2,2,1}, S_{2,3,1}, ..., S_{2,T,1}, ...)

#         df0 = 0
#         while df0 == 0:
#             if penalty_function == 'SCAD':
#                 model = GeneralizedLinearEstimator(
#                     Quadratic(),
#                     SCAD(alpha=param['alpha'], gamma=param['gamma']),
#                 )
#             elif penalty_function == 'MCP':
#                 model = GeneralizedLinearEstimator(
#                     Quadratic(),
#                     MCPenalty(alpha=param['alpha'], gamma=param['gamma']),
#                 )
#             model.fit(X, y)
#             nonzero_coef = np.where(abs(model.coef_) > 1e-4)[0]
#             df0 = len(nonzero_coef)
#             if df0 == 0:
#                 print("All regression coefficients are shrunk to 0. Increasing alpha.")
#                 # decrease alpha to avoid all zero coefficients
#                 ind = param_grid['alpha'].index(param['alpha'])
#                 ind -= 1
#                 if ind == -1:
#                     break
#                     print("Choose a smaller alpha to avoid all zero regression coefficients.")
#                 param['alpha'] = param_grid['alpha'][ind]
#                 # break
#         # df0 = (2+2*p)*p
#         print("k = ", k, ", selected covariates in regression:", nonzero_coef)
#         # compute loss
#         y_predicted = model.predict(X)
#         y_predicted = y_predicted.reshape(p, N_k, T - 1 - tau_k).transpose(1, 2, 0)
#         y_observed = y.reshape(p, N_k, T - 1 - tau_k).transpose(1, 2, 0)
#         # (y_predicted - y_observed) ** 2
#         # y_predicted[0:16]
#         # y_predicted2[0,0,:]
#         # y_predicted[2112:2112+16]
#         # y_predicted2[1,0,:]
#         loss_unnormalized_K = -np.linalg.norm(y - model.predict(X), ord=2) ** 2
#         loss_normalized_K = loss_unnormalized_K / (T-1-tau_k)
#         sigma_k = np.linalg.norm(y - model.predict(X), ord=2) ** 2 / (y.shape[0] - df0)
#         return loss_unnormalized_K, loss_normalized_K, sigma_k, df0, model, y_observed, y_predicted
#     ## END run_k

#     # loss = 0
#     loss_normalized_K = np.zeros(shape=(K))
#     loss_unnormalized_K = np.zeros(shape=(K))
#     sigma_K = np.zeros(shape=(K))
#     df_K = np.zeros(shape=(K), dtype='int32')
#     model_K = [None] * K
#     y_observed_K = {}
#     y_predicted_K = {}
#     for k in range(K):
#         loss_unnormalized_K[k], loss_normalized_K[k], sigma_K[k], df_K[k], model_K[k], y_observed_K[k], y_predicted_K[k] = run_k(k, nthread)

#     return loss_unnormalized_K, loss_normalized_K, sigma_K, df_K, model_K, y_observed_K, y_predicted_K


#%%
def goodnessofClustering(States, N, T, changepoints,Actions, g_index):
    p = States.shape[2]
    g_index = g_index.astype(int)
    K = len(set(g_index))
    changepoints = changepoints.astype(int)
    Xi = [[] for i in range(N)]
    yi = [[] for i in range(N)]
    weights = []
    df = (2+2*p)
    full_X = np.empty((0, df*K))
    full_y = np.empty((0, p))
    weights = np.empty(0)
    for i in range(int(N)):
        g = g_index.item(i)
        mat = []
        poly = PolynomialFeatures(2, interaction_only=True)
        mat_tmp = poly.fit_transform(np.vstack([Actions[i, changepoints.item(i):], States[i, changepoints.item(i):-1, range(p)]]).T)
        mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
        Xi[i] = mat
        yi[i] = States[i, changepoints.item(i)+1:,:]
        extended_x = np.zeros((mat.shape[0], df*K))
        extended_x[:, g*df: (g+1)*df] = mat
        full_X = np.append(full_X, extended_x, axis=0)
        full_y = np.append(full_y,  yi[i], axis=0)
        weights =np.append(weights, np.ones(mat.shape[0])/mat.shape[0])
    

    reg = LinearRegression(fit_intercept=False) # coef: [beta1, beta2, ..., betaK] for K clusters
    res=reg.fit(full_X, full_y, sample_weight=weights)
    # g_index_new = np.zeros(g_index.shape, dtype=int)
    
    
    # mat_list = [[] for i in range(K)]
    # y = [[] for i in range(K)]
    # Xi = [[] for i in range(N)]
    # yi = [[] for i in range(N)]
    # weights = []
    # for i in range(int(N)):
    #     g = g_index.item(i)
    #     # construct the design matrix
    #     poly = PolynomialFeatures(2, interaction_only=True)
    #     mat_tmp = poly.fit_transform(np.vstack([Actions[i, changepoints.item(i):], States[i, changepoints.item(i):-1, range(p)]]).T)
    #     mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
    #     mat_list[g].extend(mat.tolist())
    #     y[g].extend(States[i, changepoints.item(i)+1:,:].tolist())
    #     Xi[i] = mat
    #     yi[i] = States[i, changepoints.item(i)+1:,:]
    #     weights.extend([1/mat.shape[0] for r in range(p*mat.shape[0])])
    # for g in range(K):
    #     y[g] = np.array(y[g]).T.reshape(-1,1)
    #     mat_list[g] = np.vstack(np.array(mat_list[g]))
    #     mat_list[g] = np.kron(np.eye(p,dtype=int),mat_list[g])

    # X = block_diag(*mat_list)
    # y = np.vstack(y)

    # reg = LinearRegression(fit_intercept=False) # coef: [beta1, beta2, ..., betaK] for K clusters
    # res=reg.fit(X, y, sample_weight=weights)
    loss =  0
    # startTime = datetime.now()
    for i in range(N):
        # print('i', i)
        Xistack = np.kron(np.eye(K,dtype=int), Xi[i])
        yhat = res.predict(Xistack)
        err = []
        t = Xi[i].shape[0]
        for k in range(K):
            if g_index[i] == k:
                err = np.linalg.norm(yi[i] - yhat[k*t:k*t + t], ord="fro") ** 2
        loss = loss - err/t
        
    # for i in range(N):
    #     Xistack = np.kron(np.eye(K*p,dtype=int),Xi[i])
    #     yhat = res.predict(Xistack)
    #     err = []
    #     t = Xi[i].shape[0]
    #     for k in range(K):
    #         if g_index[i] == k:
    #             for j in range(p):
    #                 err.append(np.linalg.norm((yi[i][:,j].reshape([t,1]) - yhat[(j + k * p)*t :(j + k * p)*t + t]), ord=2) ** 2)
    #     err = np.sum(np.array(err))
    #     loss = loss - err/t
    # loss = loss #* np.mean(T - changepoints - 1)
    return loss

# %% clustering functions
def clustering_mean(States, N, T, K, changepoints, Actions=None, g_index=None, max_iter_gmr = 50):
    p = States.shape[2]
    X = np.zeros([N, p])
    for i in range(N):
        X[i, 0] = np.mean(States[i, changepoints[i].astype(int).item():T, :])
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    return kmeans.labels_

def clustering_marginal_dis(States, N, T, K, changepoints,Actions=None, g_index=None, max_iter_gmr = 50):
    '''
    kmeans with Mahalanobis Distance
    **cluster with size 0
    '''
    if g_index is None:
        g_index = np.random.choice(range(K), size = N)
    g_index = g_index.astype(int)
    y = [[] for i in range(K)]
    yi = [[] for i in range(N)]
    for i in range(N):
        g = g_index.item(i)
        y[g].extend(States[i, changepoints.item(i)+1:,:].tolist())
        yi[i] = States[i, changepoints.item(i)+1:,:]
    mean = np.zeros([K,1])
    var = np.zeros([K,1])
    for k in range(K):
        mean[k] = np.mean(y[k])
        var[k] = np.var(y[k])
    g_index_new = np.zeros(N, dtype=int)

    # iteration begin
    for m in range(max_iter_gmr):
        for i in range(N):
            err = []
            for k in range(K):
                err.append(np.linalg.norm((yi[i] - mean[k]), ord=2)**2/var[k])
            g_index_new[i] = err.index(min(err))
        if np.prod(g_index == g_index_new):
            break
        else:
            g_index = g_index_new
            y = [[] for i in range(K)]
            for i in range(N):
                g = g_index.item(i)
                y[g].extend(States[i, changepoints.item(i)+1:,:].tolist())
            mean = np.zeros([K,1])
            var = np.zeros([K,1])
            for k in range(K):
                mean[k] = np.mean(y[k])
                var[k] = np.var(y[k])
    return g_index

def gmr(States, N, T, K, changepoints,Actions, g_index=None, 
        max_iter_gmr = 50, logllk_loss = 0):
    '''
    mixutre regression model for normal conditional distribution
    Parameters
    ----------
    g_index : the initial group index, optional
        DESCRIPTION. The default is None.
    -------
    '''
    p = States.shape[2]
    # print('g_index',g_index)
    if g_index is None:
        g_index = np.random.choice(range(K), size = N)

    g_index = g_index.astype(int)
    changepoints = changepoints.astype(int)

    Xi = [[] for i in range(N)]
    yi = [[] for i in range(N)]
    weights = []
    df = (2+2*p)
    full_X = np.empty((0, df*K))
    full_y = np.empty((0, p))
    weights = np.empty(0)
    for i in range(int(N)):
        g = g_index.item(i)
        mat = []
        poly = PolynomialFeatures(2, interaction_only=True)
        mat_tmp = poly.fit_transform(np.vstack([Actions[i, changepoints.item(i):], States[i, changepoints.item(i):-1, range(p)]]).T)
        mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
        Xi[i] = mat
        yi[i] = States[i, changepoints.item(i)+1:,:]
        extended_x = np.zeros((mat.shape[0], df*K))
        extended_x[:, g*df: (g+1)*df] = mat
        full_X = np.append(full_X, extended_x, axis=0)
        full_y = np.append(full_y,  yi[i], axis=0)
        weights =np.append(weights, np.ones(mat.shape[0])/mat.shape[0])
    

    reg = LinearRegression(fit_intercept=False) # coef: [beta1, beta2, ..., betaK] for K clusters
    res=reg.fit(full_X, full_y, sample_weight=weights)
    g_index_new = np.zeros(g_index.shape, dtype=int)
    # iteration begin
    loss = 0
    # print('max_iter_gmr',max_iter_gmr)
    for m in range(max_iter_gmr):
        err_all = np.zeros([N, K])
        # startTime = datetime.now()
        for i in range(N):
            # print('i', i)
            Xistack = np.kron(np.eye(K,dtype=int), Xi[i])
            yhat = res.predict(Xistack)
            err = []
            t = Xi[i].shape[0]
            for k in range(K):
                # for j in range(p):
                err.append(np.linalg.norm(yi[i] - yhat[k*t:k*t + t], ord="fro") ** 2)
            err = np.array(err)
            err_all[i, :] = err.reshape(K)
            g_index_new[i] = np.where(err==min(err))[0][0]
            # if g_index_new[i]!= g_index[i]:
            #     print(i)
            loss = loss - min(err)/t

        # keep the cluster size unchanged
        if np.unique(g_index_new).shape[0] < K:
            i_tmp = -1*np.ones(len((np.setdiff1d(np.array(range(K)),np.unique(g_index_new)).tolist())))
            z=0
            for k in (np.setdiff1d(np.array(range(K)),np.unique(g_index_new)).tolist()):
                # print('i_tmp[',z,']',i_tmp[z],np.where(err_all[:,k] == min(err_all[:,k])))
                smallest = 0
                assign_ind = np.where(err_all[:,k] == min(err_all[:,k]))[0][0]
                # print('err_all[:,k]',err_all[:,k],'np',np.where(err_all[:,k] == min(err_all[:,k]))[0], ", assign_ind", assign_ind)
                while (assign_ind in i_tmp) or (np.sum(g_index_new == g_index_new[assign_ind])==1):
                    smallest = smallest+1
                    # print(smallest)
                    assign_ind = np.where(err_all[:,k] == np.partition(err_all[:,k], smallest)[smallest])[0][0]
                t = Xi[assign_ind].shape[0]
                loss = loss + err_all[assign_ind, g_index_new[assign_ind]]/t - err_all[assign_ind,k]/t
                g_index_new[assign_ind] = k
                i_tmp[z]=assign_ind
                z = z+1
        # print('loss',loss, ', T',T, ', np.mean(T - changepoints - 1)',np.mean(T - changepoints - 1))
        loss = loss #* np.mean(T - changepoints - 1)
        # print('K', K, 'loss',loss)
        if np.prod(g_index == g_index_new) or m == max_iter_gmr - 1:
            break
        else: # not np.prod(g_index == g_index_new):
            loss = 0
            g_index = g_index_new.copy()
            full_X = np.empty((0, df*K))
            full_y = np.empty((0, p))
            weights = np.empty(0)
            for i in range(int(N)):
                g = g_index.item(i)
                mat = []
                poly = PolynomialFeatures(2, interaction_only=True)
                mat_tmp = poly.fit_transform(np.vstack([Actions[i, changepoints.item(i):], States[i, changepoints.item(i):-1, range(p)]]).T)
                mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
                Xi[i] = mat
                yi[i] = States[i, changepoints.item(i)+1:,:]
                extended_x = np.zeros((mat.shape[0], df*K))
                extended_x[:, g*df: (g+1)*df] = mat
                full_X = np.append(full_X, extended_x, axis=0)
                full_y = np.append(full_y,  yi[i], axis=0)
                weights =np.append(weights, np.ones(mat.shape[0])/mat.shape[0])
                
            res=reg.fit(full_X, full_y, sample_weight=weights)
            
    if logllk_loss:
        loss = np.sum(norm.pdf(err_all, scale=np.sqrt(np.linalg.norm(err_all, ord="fro")**2/err_all.size)))
    return g_index_new, loss

def gmr_old(States, N, T, K, changepoints,Actions, g_index=None, max_iter_gmr = 50):
    '''
    mixutre regression model for normal conditional distribution
    Parameters
    ----------
    g_index : the initial group index, optional
        DESCRIPTION. The default is None.
    -------
    '''
    p = States.shape[2]
    # print('g_index',g_index)
    if g_index is None:
        g_index = np.random.choice(range(K), size = N)
        # print(N)
    # g_index = np.append(np.zeros(int(N/2)), np.ones(int(N/2)))

    g_index = g_index.astype(int)
    # print('gmr: beginning g_index', g_index)
    changepoints = changepoints.astype(int)
    # print('changepoints',changepoints)

    mat_list = [[] for i in range(K)]
    y = [[] for i in range(K)]
    Xi = [[] for i in range(N)]
    yi = [[] for i in range(N)]
    weights = []
    for i in range(int(N)):
        # print('i', i)
        g = g_index.item(i)
        # print('g', g)
        # construct the design matrix
        mat = []
        poly = PolynomialFeatures(2, interaction_only=True)
        mat_tmp = poly.fit_transform(np.vstack([Actions[i, changepoints.item(i):], States[i, changepoints.item(i):-1, range(p)]]).T)
        mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
        mat_list[g].extend(mat.tolist())
        y[g].extend(States[i, changepoints.item(i)+1:,:].tolist())
        Xi[i] = mat
        yi[i] = States[i, changepoints.item(i)+1:,:]
        # print('mat.shape[0]', mat.shape[0],'cp',changepoints.item(i),'1/mat.shape[0]',1/mat.shape[0])
        weights.extend([1/mat.shape[0] for r in range(p*mat.shape[0])])
    
    for g in range(K):
        y[g] = np.array(y[g]).T.reshape(-1,1)
        mat_list[g] = np.vstack(np.array(mat_list[g]))
        mat_list[g] = np.kron(np.eye(p,dtype=int),mat_list[g])

    X = block_diag(*mat_list)
    y = np.vstack(y)

    reg = LinearRegression(fit_intercept=False) # coef: [beta1, beta2, ..., betaK] for K clusters
    res=reg.fit(X, y, sample_weight=weights)
    g_index_new = np.zeros(g_index.shape, dtype=int)
    # iteration begin
    loss = 0
    # print('max_iter_gmr',max_iter_gmr)
    for m in range(max_iter_gmr):
        err_all = np.zeros([N, K])
        # startTime = datetime.now()
        for i in range(N):
            # print('i', i)
            Xistack = np.kron(np.eye(K*p,dtype=int),Xi[i])
            yhat = res.predict(Xistack)
            err = [[] for i in range(K)]
            t = Xi[i].shape[0]
            for k in range(K):
                for j in range(p):
                    err[k].append(np.linalg.norm((yi[i][:,j].reshape([t,1]) - yhat[(j + k * p)*t:(j + k * p)*t + t]), ord=2) ** 2)
            err = np.sum(np.array(err), axis=1)
            err_all[i, :] = err.reshape(K)
            g_index_new[i] = np.where(err==min(err))[0][0]
            # print('minerr', min(err))
            loss = loss - 1 * min(err)/t
            # print('loss', loss)
        # keep the cluster size unchanged
        if np.unique(g_index_new).shape[0] < K:
            i_tmp = -1*np.ones(len((np.setdiff1d(np.array(range(K)),np.unique(g_index_new)).tolist())))
            z=0
            for k in (np.setdiff1d(np.array(range(K)),np.unique(g_index_new)).tolist()):
                # print('i_tmp[',z,']',i_tmp[z],np.where(err_all[:,k] == min(err_all[:,k])))
                smallest = 0
                assign_ind = np.where(err_all[:,k] == min(err_all[:,k]))[0][0]
                # print('err_all[:,k]',err_all[:,k],'np',np.where(err_all[:,k] == min(err_all[:,k]))[0], ", assign_ind", assign_ind)
                while (assign_ind in i_tmp) or (np.sum(g_index_new == g_index_new[assign_ind])==1):
                    smallest = smallest+1
                    # print(smallest)
                    assign_ind = np.where(err_all[:,k] == np.partition(err_all[:,k], smallest)[smallest])[0]
                loss = loss + err_all[assign_ind,g_index_new[assign_ind]] - err_all[assign_ind,k]
                g_index_new[assign_ind] = k
                i_tmp[z]=assign_ind
                z = z+1
        # print('loss',loss, ', T',T, ', np.mean(T - changepoints - 1)',np.mean(T - changepoints - 1))
        loss = loss * np.mean(T - changepoints - 1)
        # print('loss',loss)
        if np.prod(g_index == g_index_new) or m == max_iter_gmr - 1:
            # print('break')
            break
        else: # not np.prod(g_index == g_index_new):
            loss = 0
            g_index = g_index_new.copy()
            mat_list = [[] for i in range(K)]
            y = [[] for i in range(K)]
            weights = []
            # startTime = datetime.now()
            for i in range(int(N)):
                # print('g_index.shape',g_index.shape)
                # print('i',i)
                g = g_index.item(i)
                # construct the design matrix
                mat = []
                poly = PolynomialFeatures(2, interaction_only=True)
                mat_tmp = poly.fit_transform(np.vstack([Actions[i, changepoints.item(i):], States[i, changepoints.item(i):-1, range(p)]]).T)
                mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
                mat_list[g].extend(mat.tolist())
                y[g].extend(States[i, changepoints.item(i)+1:,:].tolist())
                weights.extend([1/mat.shape[0] for r in range(p*mat.shape[0])])
            for g in range(K):
                # print('g',g)
                y[g] = np.array(y[g]).T.reshape(-1,1)
                mat_list[g] = np.vstack(np.array(mat_list[g]))
                mat_list[g] = np.kron(np.eye(p,dtype=int),mat_list[g])

            X = block_diag(*mat_list)
            y = np.vstack(y)
            res=reg.fit(X, y, sample_weight=weights)
    return g_index_new, loss

def tuneK_wrap(K_list, States, N, T, changepoints,Actions,
               g_index=None,max_iter_gmr=20, C=5, is_tunek_wrap_parallel=1,Kl_fun='logN', ic_T_dynamic=0):
    '''
    Trun K in each iteration from K_list

    '''
    def run_K(K, States, N, T, changepoints, Actions, g_index, max_iter_gmr=max_iter_gmr, C=C):
        g_index = g_index if K == len(np.unique(g_index)) else None
        # g_index, loss = clustering(States=States, N=N, T=T, K=K, changepoints=changepoints,g_index = g_index,
        #                              example=example, Actions=Actions)
        g_index, loss = gmr(States, N, T, K, changepoints, Actions, g_index=None, max_iter_gmr=max_iter_gmr)
        result = namedtuple("result", ["IC", "g_index", "loss"])
        return result(ut.IC(loss=loss, changepoints=changepoints, g_index=g_index, N=N, T=T, K=K, C=C, Kl_fun=Kl_fun, ic_T_dynamic=ic_T_dynamic)
                      , g_index, loss)
    IC_max = None
    best_g_index = None
    best_loss = None
    if is_tunek_wrap_parallel:
        nthread = len(K_list)
        res = Parallel(n_jobs=nthread)(delayed(run_K)(K, States, N, T, changepoints, Actions, g_index, max_iter_gmr, C) for K in K_list)
        for K in K_list:
            if best_g_index is None:
                best_g_index = res[K_list.index(K)].g_index
                IC_max = res[K_list.index(K)].IC
                best_loss = res[K_list.index(K)].loss
            elif IC_max < res[K_list.index(K)].IC:
                best_g_index = res[K_list.index(K)].g_index
                IC_max = res[K_list.index(K)].IC
                best_loss = res[K_list.index(K)].loss
    else:
        for K in K_list:
            print('K', K)
            tmp = run_K(K, States, N, T, changepoints, Actions, g_index, max_iter_gmr, C)
            # res.append(tmp)
            if best_g_index is None:
                best_g_index = tmp.g_index
                IC_max = tmp.IC
                best_loss = tmp.loss
            elif IC_max < tmp.IC:
                best_g_index = tmp.g_index
                IC_max = tmp.IC
                best_loss = tmp.loss
    return best_g_index, best_loss
    

#%% changepoint detection functions
def changemean_detect(g_index, States, N, T, kappa_max,kappa_min,kappa_interval, epsilon,Actions=None, cusum_forward=None,
                      cusum_backward=None, C1=None, C2=None, alpha = 0.05, df=5, nthread=0):
    '''
    detect change in means
    need adaptations for kappa
    '''
    K = len(set(g_index))
    tauk = np.zeros(K)
    changepoints = np.zeros(N).reshape(-1,1)
    maxcusum_list = np.zeros(K)
    for k in range(K):
        for kappa in range(kappa_min, kappa_max):
            maxcusum = 0
            def run_one(u):
                return(np.sqrt((T-u)*(u-T+kappa)/(kappa**2))*np.sum(g_index == k)
                             * np.linalg.norm(
                                 np.dot((g_index == k).reshape(1, N),
                                        (cusum_forward[:, u] - cusum_backward[:, u+1]).reshape(N,1)), ord = 2))
            if nthread !=0:
                res = Parallel(n_jobs = nthread)(delayed(run_one)(u) for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1))
                tauk[k] =int(T-epsilon*T)-1- res.index(max(res))
                maxcusum = np.max(res)
            else:
                for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1):
                    cusum_tmp = run_one(u)
                    if u == int(T-epsilon*T)-1:
                        maxcusum = cusum_tmp
                        maxcusum_list[k] = maxcusum
                        tauk[k] = u
                    elif maxcusum <= cusum_tmp:
                        # print('** maxcusum',maxcusum)
                        maxcusum = cusum_tmp
                        maxcusum_list[k] = maxcusum
                        tauk[k] = u
            if maxcusum < C1*(kappa * np.sum(g_index == k))**(-1*C2) * (np.log10(N * T)):
                tauk[k] = int(T - kappa + epsilon*T) + 1
            changepoints[(g_index == k), :] = tauk[k]
            changepoints = changepoints.astype(int)
    return [changepoints, maxcusum_list, tauk]

def changemarginal_detect(g_index, States, N, T, kappa_max, kappa_min,kappa_interval, epsilon, Actions=None,
                              cusum_forward=None, cusum_backward=None, C1=None,
                              C2=None, alpha = 0.05, df = 5, nthread=0):
    '''
    detect change in marginal distribution
    need adaptations for kappa
    '''
    K = len(set(g_index))
    # print("K",K)
    tauk = np.zeros(K)
    changepoints = np.zeros(N).reshape(-1,1)
    maxcusum_list = np.zeros(K)
    def run_one(u):
       y1 = []
       y2 = []
       y = []
       for i in range(int(N)):
           if g_index[i] == k:
               y1.append(States[i, :u+1,:])
               y2.append(States[i, u+1:,:])
               y.append(States[i, :,:])
       y1 = np.vstack(np.array(y1))
       y2 = np.vstack(np.array(y2))
       y = np.vstack(np.array(y))
       var0 = np.var(y)
       var1 = (np.var(y1) * np.size(y1)+np.var(y2)*np.size(y2))/(np.sum(g_index == k))#*T
       mean0 = np.mean(y)
       mean1 = np.mean(y1)
       mean2 = np.mean(y2)
       L0 = norm.pdf(y, loc = mean0, scale = np.sqrt(var0))
       L1 = np.vstack([norm.pdf(y1, loc = mean1, scale = np.sqrt(var1)),norm.pdf(y2, loc = mean2, scale = np.sqrt(var1))])
       L0[np.where(L0 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
       L1[np.where(L1 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
       logL0 = np.sum(np.log(L0))
       logL1 = np.sum(np.log(L1))
       cusum_tmp =-(logL0 - logL1)/(kappa * np.sum(g_index == k)) #-2*(logL0 - logL1)
       return cusum_tmp

    for k in range(K):
        for kappa in range(kappa_min, kappa_max):
            maxcusum = 0
            if nthread !=0: # Parallel
                res = Parallel(n_jobs=nthread)(delayed(run_one)(u) for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1))
                tauk[k] = int(T-epsilon*T)-1- res.index(np.max(res))
                maxcusum= np.max(res)
            else: # do not Parallel
                cusum_list = []
                for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1):
                    cusum_tmp = run_one(u)
                    cusum_list.append(cusum_tmp)
                    if u == int(T-epsilon*T)-1:
                        maxcusum = cusum_tmp
                        maxcusum_list[k] = maxcusum
                        tauk[k] = u
                    elif maxcusum <= cusum_tmp:
                        # print('** maxcusum',maxcusum)
                        maxcusum = cusum_tmp
                        maxcusum_list[k] = maxcusum
                        tauk[k] = u
            if maxcusum < (C1*(kappa * np.sum(g_index == k))**(-1*C2) * np.log10(N * T))**2:#chi2.ppf(1-alpha, df):
                tauk[k] = int(T - kappa + epsilon*T) + 1
            changepoints[(g_index == k), :] = tauk[k]
    changepoints = changepoints.astype(int)
    return [changepoints, tauk]

def changedistribution_detect2(g_index, States, N, T, kappa_max, kappa_min, kappa_interval, 
             epsilon, Actions=None, cusum_forward=None, cusum_backward=None, C1=None,
        C2=None, alpha = 0.01, df = 5, nthread=3, threshold_type="maxcusum", 
        nthread_B= None, B = 2000, is_cp_parallel=1, save_path=None):
    K = len(set(g_index))
    print('g_index',g_index,', K',K)
    p = States.shape[2]
    tauk = np.zeros(K)
    changepoints = np.zeros(N).reshape(-1,1)
    # maxcusum_list = np.zeros(K)
    kappa_list = np.arange(kappa_min, min(kappa_max, T-1), step=kappa_interval, dtype=np.int32)
    #%%
    def run_one(u, seq, k, kappa):
        States_current = States[:,seq,:]
        States_next = States[:, seq+1, :]
        Actions_current = Actions[:,seq]
        X1 = []
        y1 = []
        X2 = []
        y2 = []
        X = []
        y = []
        for i in range(int(N)):
            if g_index[i] == k:
                poly = PolynomialFeatures(2, interaction_only=True)
                # print('Actions_current[i, :u]',Actions_current[i, :u])
                mat_tmp = poly.fit_transform(np.vstack([Actions_current[i, :u], States_current[i, :u, range(p)]]).T)
                mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
                X1.append(mat)
                y1.append(States_next[i, :u,:])

                poly = PolynomialFeatures(2, interaction_only=True)
                mat_tmp = poly.fit_transform(np.vstack([Actions_current[i, u:], States_current[i, u:, range(p)]]).T)
                mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
                X2.append(mat)
                y2.append(States_next[i, u:,:])

                poly = PolynomialFeatures(2, interaction_only=True)
                mat_tmp = poly.fit_transform(np.vstack([Actions_current[i, :], States_current[i, :, range(p)]]).T)
                mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
                X.append(mat)
                y.append(States_next[i, :,:])

        X1 = np.vstack(np.array(X1))
        X1 = np.kron(np. eye(p,dtype=int),X1)
        y1 = np.vstack(np.array(y1)).T.reshape(-1, 1)
        X2 = np.vstack(np.array(X2))
        X2 = np.kron(np.eye(p,dtype=int),X2)
        y2 = np.vstack(np.array(y2)).T.reshape(-1, 1)
        X = np.vstack(np.array(X))
        X = np.kron(np.eye(p,dtype=int),X)
        y = np.vstack(np.array(y)).T.reshape(-1, 1)
        reg0 = LinearRegression(fit_intercept=False)
        res0 = reg0.fit(X, y)
        reg1 = LinearRegression(fit_intercept=False)
        res1=reg1.fit(X1, y1)
        reg2 = LinearRegression(fit_intercept=False)
        res2=reg2.fit(X2, y2)
        # print(res1.coef_, res2.coef_)

        var0 = np.linalg.norm(y - res0.predict(X), ord=2) ** 2 / y.shape[0]
        var0 = np.sum((y - res0.predict(X))** 2)  / y.shape[0]
        # weigh the variance of each segment by their sample sizes: (# people) * (# time points) * (# dimension of States)
        var1 = (np.linalg.norm(y1 - res1.predict(X1), ord=2) ** 2
                + np.linalg.norm(y2 - res2.predict(X2), ord=2) ** 2) / y.shape[0]
        if var0 <  np.finfo(np.float64).tiny:
            var0 =  np.finfo(np.float64).tiny
        if var1 <  np.finfo(np.float64).tiny:
            var1 =  np.finfo(np.float64).tiny
           
        mean0 = res0.predict(X)
        mean1 = res1.predict(X1)
        mean2 = res2.predict(X2)

        L0 = norm.pdf(y, loc = mean0, scale = np.sqrt(var0))
        L1 = np.vstack([norm.pdf(y1, loc = mean1, scale = np.sqrt(var1)),
                        norm.pdf(y2, loc = mean2, scale = np.sqrt(var1))])
        L0[np.where(L0 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
        L1[np.where(L1 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
        logL0 = np.sum(np.log(L0))
        logL1 = np.sum(np.log(L1))
        cusum_tmp = -1*(logL0 - logL1) #/(np.sum(g_index== k)*kappa)
        return cusum_tmp
    #%%
    def run_k(k, nthread_B=nthread_B):
        
        for kappa in kappa_list:
            print('====== kappa',kappa)
            is_cp_found=1
            maxcusum = 0
            seq = np.arange(T-kappa-1, T-1)
            u_list =  range(int(-epsilon*T+kappa), int(epsilon*T-1), -1)
            if nthread >=1: # Parallel
                # startTime = datetime.now()
                res = Parallel(n_jobs=nthread)(delayed(run_one)(u, seq, k, kappa) for u in u_list)
                # print('np.max(res)',np.max(res),', ', np.argmax(res))
                # print('one cusum: ', datetime.now() - startTime)
                sys.stdout.flush()
                maxcusum= np.max(res)
            else: # do not Parallel
                for u in u_list:
                    print('u', u)
                    cusum_tmp = run_one(u, seq, k, kappa)
                    print('cusum', cusum_tmp)
                    if u == int(T-epsilon*T)-1:
                        maxcusum = cusum_tmp
                        # maxcusum_list[k] = maxcusum
                        tau = u
                    elif maxcusum <= cusum_tmp:
                        print('** maxcusum',maxcusum)
                        maxcusum = cusum_tmp
                        # maxcusum_list[k] = maxcusum
                        tau = u
            if threshold_type == "maxcusum":
                # startTime = datetime.now()
                # print('nthread', nthread)
                threshold = ut.estimate_threshold(np.sum(g_index == k), 
                                                  kappa, df, nthread=nthread,
                                                  B = B, alpha = alpha, save_path=save_path)[0]
                # print('threshold time: ', datetime.now() - startTime)
            elif threshold_type == "chi2":
                threshold, stats = ut.chi2_threshold(np.sum(g_index == k), len(u_list), df, nthread=nthread, B = B, alpha = alpha) 
            elif threshold_type == "permutation":
                if nthread_B is None:
                    nthread_B = nthread
                shift_index = []
                for i in range(B):
                    random.shuffle(seq)
                    tmp = seq.copy()
                    shift_index.append(tmp)
                # startTime = datetime.now()
                sample_stat = Parallel(n_jobs=nthread_B)(delayed(run_one)(u, seq, k, kappa) for u in range(int(-epsilon*T+kappa), int(epsilon*T-1), -1)
                                                       for seq in shift_index)
                # print('threshold time: ', datetime.now() - startTime)
                sys.stdout.flush()
                sample_stat = np.max(np.array(sample_stat).reshape([B, -1]), axis = 1)
                threshold = np.percentile(sample_stat, (1 - alpha)*100)
            print("maxcusum =", maxcusum)
            print("threshold =", threshold)
            sys.stdout.flush()
            if maxcusum < threshold:
                tau = 0
                is_cp_found=0
            else:
                if np.where(kappa_list == kappa)[0][0] == 0:
                    tau = T-1-kappa_list[np.where(kappa_list == kappa)]
                else: 
                    # print('T',T, 'T-1-kappa_list[np.where(kappa_list == kappa)[0][0] - 1]',T-1-kappa_list[np.where(kappa_list == kappa)[0][0] - 1])
                    tau = T-1-kappa_list[np.where(kappa_list == kappa)[0][0] - 1]
            if is_cp_found:
                print('!!!!!!! found!!!! tauk[k]',tau)
                break
        return tau
    #%%
    startTime = datetime.now()
    if is_cp_parallel:
        tauk = Parallel(n_jobs=K)(delayed(run_k)(k) for k in range(K))
        for k in range(K):
            # print('g_index', g_index)
            changepoints[(g_index == k), :] = tauk[k]
    else:
        for k in range(K):
            tauk[k] = run_k(k)
            changepoints[(g_index == k), :] = tauk[k]
           # print('k',k)
    # print('is_cp_parallel', is_cp_parallel, ', finish time', datetime.now() - startTime, ', tauk', tauk)
    changepoints = changepoints.astype(int)
    sys.stdout.flush()
    return [changepoints, tauk]
    
#%% add penalty to the regression
# def changedistribution_detect_sparse(g_index, States, N, T, kappa_max, kappa_min, kappa_interval,
#         param_grid, nfold = 5, penalty_function = 'SCAD', select_param_interval = 5,
#         epsilon=0.1, Actions=None, cusum_forward=None, cusum_backward=None, C1=None,
#         C2=None, alpha = 0.01, df = 5, nthread=3, threshold_type="maxcusum",
#         nthread_B= None, B = 2000, is_cp_parallel=1, seed=0, break_early=1):

#     # %% compute test statistic for each u
#     def run_one(u, States_current, States_next, Actions_current,
#                 penalty_function, param, nonzero_coef, logL0, kappa):
#         p = States_current.shape[2]
#         if penalty_function == 'SCAD':
#             model = GeneralizedLinearEstimator(
#                 Quadratic(),
#                 SCAD(alpha=param['alpha'], gamma=param['gamma']),
#             )
#         elif penalty_function == 'MCP':
#             model = GeneralizedLinearEstimator(
#                 Quadratic(),
#                 MCPenalty(alpha=param['alpha'], gamma=param['gamma']),
#             )

#         ## next we will restrict the covariates in X1 and X2 to be those selected above
#         N_k = States_current.shape[0]
#         X1 = []
#         y1 = []
#         X2 = []
#         y2 = []
#         for i in range(N_k):
#             poly = PolynomialFeatures(2, interaction_only=True)
#             mat_tmp = poly.fit_transform(np.vstack([Actions_current[i, :u], States_current[i, :u, range(p)]]).T)
#             mat = np.delete(mat_tmp, np.s_[2 + p * 2:mat_tmp.shape[1]], 1)
#             X1.append(mat)
#             y1.append(States_next[i, :u, :])

#             poly = PolynomialFeatures(2, interaction_only=True)
#             mat_tmp = poly.fit_transform(np.vstack([Actions_current[i, u:], States_current[i, u:, range(p)]]).T)
#             mat = np.delete(mat_tmp, np.s_[2 + p * 2:mat_tmp.shape[1]], 1)
#             X2.append(mat)
#             y2.append(States_next[i, u:, :])

#         X1 = np.vstack(np.array(X1))
#         X1 = np.kron(np.eye(p, dtype=int), X1)
#         X1 = X1[:, nonzero_coef]
#         X1 = sparse.csr_matrix(X1)
#         y1 = np.vstack(np.array(y1)).T.reshape(-1, 1)
#         X2 = np.vstack(np.array(X2))
#         X2 = np.kron(np.eye(p, dtype=int), X2)
#         X2 = X2[:, nonzero_coef]
#         X2 = sparse.csr_matrix(X2)
#         y2 = np.vstack(np.array(y2)).T.reshape(-1, 1)

#         reg1 = LinearRegression(fit_intercept=False)
#         res1 = reg1.fit(X1, y1)
#         reg2 = LinearRegression(fit_intercept=False)
#         res2 = reg2.fit(X2, y2)

#         # weigh the variance of each segment by their sample sizes: (# people) * (# time points) * (# dimension of States)
#         var1 = (np.linalg.norm(y1 - res1.predict(X1), ord=2) ** 2
#                 + np.linalg.norm(y2 - res2.predict(X2), ord=2) ** 2) / (y1.shape[0] + y2.shape[0])
#         if var1 < np.finfo(np.float64).tiny:
#             var1 = np.finfo(np.float64).tiny
#         mean1 = res1.predict(X1)
#         mean2 = res2.predict(X2)
#         L1 = np.vstack([norm.pdf(y1, loc=mean1, scale=np.sqrt(var1)), norm.pdf(y2, loc=mean2, scale=np.sqrt(var1))])
#         L1[np.where(L1 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
#         logL1 = np.sum(np.log(L1))
#         cusum_tmp = -1 * (logL0 - logL1) / (States_current.shape[0] * kappa)
#         return cusum_tmp
#     ### END run_one

#     #%% compute test statistic for each cluster k, taking maximum over u
#     def run_k(k, nthread_B=nthread_B):
#         States_k = States_s[(g_index == k), :, :]
#         Actions_k = Actions[(g_index == k), :]
#         p_values_by_kappa = []
#         dfs = []
#         # indicate whether we have encountered the first cp
#         first_cp = 1
#         kappa_index = 0
#         for kappa in kappa_list:
#             if int(2 * epsilon * T) >= kappa:
#                 print("Error: We require (2 * epsilon * T) < kappa")
#                 break
#             is_cp_found=1
#             maxcusum = 0
#             u_list = np.arange(int(epsilon*T), int(-epsilon*T+kappa)+1, step=1)
#             seq = np.arange(T-kappa-1, T-1)
#             States_after_kappa = States_k[:, (T-kappa-1):, :]
#             Actions_after_kappa = Actions_k[:, (T-kappa-1):]

#             # select tuning parameters
#             if kappa_index % select_param_interval == 0:
#                 if States_after_kappa.shape[0] >= 10:
#                     cv_result = grid_search_cv(States_after_kappa, Actions_after_kappa, param_grid, nfold, nthread_B, penalty_function, seed)
#                     param = cv_result['selected_param']
#                 else:
#                     param = {'alpha': max(param_grid['alpha']), 'gamma': max(param_grid['gamma'])}
#                 kappa_index = 0
#             kappa_index += 1

#             #%% fit model on all data from T-kappa to T
#             States_current = States_after_kappa[:, :-1, :]
#             States_next = States_after_kappa[:, 1:, :]
#             Actions_current = Actions_after_kappa
#             X = []
#             y = []
#             for i in range(States_current.shape[0]):
#                 poly = PolynomialFeatures(2, interaction_only=True)
#                 mat_tmp = poly.fit_transform(
#                     np.vstack([Actions_current[i, :], States_current[i, :, range(p)]]).T)
#                 mat = np.delete(mat_tmp, np.s_[2 + p * 2:mat_tmp.shape[1]], 1)
#                 X.append(mat)
#                 y.append(States_next[i, :, :])
#             # perform regression
#             X = np.vstack(np.array(X))
#             X = np.kron(np.eye(p, dtype=int), X)
#             X = sparse.csr_matrix(X)
#             y = np.vstack(np.array(y)).T.reshape(-1, )

#             df0 = 0
#             while df0 == 0:
#                 if penalty_function == 'SCAD':
#                     model = GeneralizedLinearEstimator(
#                         Quadratic(),
#                         SCAD(alpha=param['alpha'], gamma=param['gamma']),
#                     )
#                 elif penalty_function == 'MCP':
#                     model = GeneralizedLinearEstimator(
#                         Quadratic(),
#                         MCPenalty(alpha=param['alpha'], gamma=param['gamma']),
#                     )
#                 model.fit(X, y)
#                 nonzero_coef = np.where(abs(model.coef_) > 1e-4)[0]
#                 df0 = len(nonzero_coef)
#                 if df0 == 0:
#                     print("All regression coefficients are shrunk to 0. Increasing alpha.")
#                     # decrease alpha to avoid all zero coefficients
#                     ind = param_grid['alpha'].index(param['alpha'])
#                     ind -= 1
#                     if ind == -1:
#                         break
#                         print("Choose a smaller alpha to avoid all zero regression coefficients.")
#                     param['alpha'] = param_grid['alpha'][ind]
#                     # break
#             # df0 = (2+2*p)*p
#             print("k = ", k, ", selected covariates in regression:", nonzero_coef)
#             # compute likelihood ratio stat
#             var0 = np.linalg.norm(y - model.predict(X), ord=2) ** 2 / y.shape[0]
#             if var0 < np.finfo(np.float64).tiny:
#                 var0 = np.finfo(np.float64).tiny
#             mean0 = model.predict(X)
#             L0 = norm.pdf(y, loc=mean0, scale=np.sqrt(var0))
#             logL0 = np.sum(np.log(L0))


#             #%% compute test statistic for each u
#             if nthread != 1:  # Parallel
#                 res = Parallel(n_jobs=nthread)(delayed(run_one)(u, States_current, States_next, Actions_current,
#                         penalty_function, param, nonzero_coef, logL0, kappa) for u in u_list)
#                 maxcusum = np.max(res)
#             else: # do not Parallel
#                 for u in u_list:
#                     cusum_tmp = run_one(u, States_current, States_next, Actions_current,
#                         penalty_function, param, nonzero_coef, logL0, kappa)
#                     if u == int(T - epsilon * T) - 1:
#                         maxcusum = cusum_tmp
#                         tau = u
#                     elif maxcusum <= cusum_tmp:
#                         maxcusum = cusum_tmp
#                         tau = u
#             if threshold_type == "maxcusum":
#                 startTime = datetime.now()
#                 threshold, sample_stat = ut.estimate_threshold(N=np.sum(g_index == k), kappa=kappa, df=df0, nthread=nthread, B = B, alpha = alpha, seed=seed)
#                 sys.stdout.flush()
#             elif threshold_type == "permutation":
#                 if nthread_B is None:
#                     nthread_B = nthread
#                 shift_index = []
#                 for i in range(B):
#                     random.shuffle(seq)
#                     tmp = seq.copy()
#                     shift_index.append(tmp)
#                 sample_stat = Parallel(n_jobs=nthread_B)(delayed(run_one)(u, seq, k, kappa) for u in range(int(-epsilon*T+kappa), int(epsilon*T-1), -1)
#                                                        for seq in shift_index)
#                 sys.stdout.flush()
#                 sample_stat = np.max(np.array(sample_stat).reshape([B, -1]), axis = 1)
#                 threshold = np.percentile(sample_stat, (1 - alpha)*100)
#             p_value = np.mean(maxcusum < sample_stat)
#             p_values_by_kappa.append(p_value)
#             dfs.append(df0)
#             print('k =', k, "\nkappa     =", kappa)
#             print("maxcusum  =", round(maxcusum, 5))
#             print("threshold =", round(threshold, 5))
#             print("p_value   =", round(p_value, 5))
#             print("df        =", df0)
#             sys.stdout.flush()
#             if maxcusum < threshold:
#                 tau = 0
#                 is_cp_found=0
#             else:
#                 if np.where(kappa_list == kappa)[0][0] == 0:
#                     tau = T-1-kappa_list[np.where(kappa_list == kappa)]
#                     tau = tau[0]
#                 else:
#                     tau = T-1-kappa_list[np.where(kappa_list == kappa)[0][0] - 1]
#             # print("good1")
#             if is_cp_found:
#                 if first_cp:
#                     tau_final = tau
#                     p_value_final = p_value
#                 first_cp = 0
#                 if break_early:
#                     break
#         # print("good2")
#         if 'tau_final' in vars() or 'tau_final' in globals():
#             return [tau_final, p_value_final, p_values_by_kappa, dfs]
#         else:
#             return [tau, p_value, p_values_by_kappa, dfs]
#     ### END run_k


#     #%%
#     K = len(set(g_index))
#     p = States.shape[2]
#     tauk = np.zeros(K)
#     changepoints = np.zeros(N).reshape(-1,1)
#     p_values_cp = np.zeros(K)
#     p_values_cp[:] = np.nan
#     p_values = np.zeros(shape = (K, kappa_max - kappa_min))
#     p_values[:] = np.nan
#     dfs = np.zeros(shape = (K, kappa_max - kappa_min))
#     dfs[:] = np.nan
#     # maxcusum_list = np.zeros(K)
#     kappa_list = np.arange(kappa_min, kappa_max, step=kappa_interval, dtype=np.int32)


#     # normalize states by each cluster during change point detection
#     States_s = copy(States)
#     def transform(x):
#         return (x - np.mean(x)) / np.std(x)
#     for k in range(K):
#         for pp in range(p):
#             States_s[(g_index == k), :, pp] = transform(States[(g_index == k), :, pp])


#     startTime = datetime.now()
#     if is_cp_parallel:
#         # print("running run_k")
#         out = Parallel(n_jobs=K)(delayed(run_k)(k) for k in range(K))
#         out1 = [item[0] for item in out]
#         out2 = [item[1] for item in out]
#         p_values_by_kappa = [item[2] for item in out]
#         dfs_out = [item[3] for item in out]
#         # out1, out2, p_values_by_kappa, dfs_out = zip(*Parallel(n_jobs=K)(delayed(run_k)(k) for k in range(K)))
#         # print("p_values_by_kappa =", p_values_by_kappa)
#         # tauk = out1[0][0]
#         for k in range(K):
#             tauk[k] = out1[k]
#             p_values_cp[k] = out2[k]
#             changepoints[(g_index == k), :] = tauk[k]
#             p_values[k, :len(p_values_by_kappa[k])] = p_values_by_kappa[k]
#             dfs[k, :len(dfs_out[k])] = dfs_out[k]  # p_value_k[k]
#     else:
#         for k in range(K):
#             tauk[k], p_values_cp[k], p_values_by_kappa, dfs_out = run_k(k)
#             changepoints[(g_index == k), :] = tauk[k]
#             p_values[k, :len(p_values_by_kappa[k])] = p_values_by_kappa[k]
#             dfs[k, :len(dfs_out[k])] = dfs_out[k]  # p_value_k[k]

#     print("changepoint       =", tauk)
#     print("p_values_by_kappa =", p_values_by_kappa)
#     # print("p_values_cp       =", p_values_cp)
#     print("dfs               =", dfs)
#           # print('k',k)
#     # print('is_cp_parallel', is_cp_parallel, ', finish time', datetime.now() - startTime, ', tauk', tauk)
#     changepoints = changepoints.astype(int)
#     sys.stdout.flush()
#     return [changepoints, tauk, p_values_cp, p_values, dfs]


def permutation_test(States_ori, Actions_ori, g_index, k, u, nthread_B=1):
    N = States_ori.shape[0]
    T = States_ori.shape[1]
    p = States_ori.shape[2]
    
    shift_index = []
    shift_index.append(np.array(range(T-1)))
    for i in range(T-2):
        shift_index.append(np.roll(shift_index[i], 1))
    def run_one(seq):
        # States_current = States_ori[:, :-1, :]
        # States_next = States_ori[:, 1:, :]
        # seq = np.array(range(T-1))
        # seq = np.random.permutation(range(T-1))
        States_current = States_ori[:,seq,:]
        # print('States_current.shape',States_current.shape)
        States_next = States_ori[:, seq+1, :]
        Actions = Actions_ori[:,seq]
        # print('Actions',Actions.shape)
        X1 = []
        y1 = []
        X2 = []
        y2 = []
        X = []
        y = []
        # N_k = sum(g_index == k)
        for i in range(int(N)):
            if g_index[i] == k:
                poly = PolynomialFeatures(2, interaction_only=True)
                mat_tmp = poly.fit_transform(np.vstack([Actions[i, :u], States_current[i, :u, range(p)]]).T)
                mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
                X1.append(mat)
                y1.append(States_next[i, :u,:])

                poly = PolynomialFeatures(2, interaction_only=True)
                mat_tmp = poly.fit_transform(np.vstack([Actions[i, u:], States_current[i, u:, range(p)]]).T)
                mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
                X2.append(mat)
                y2.append(States_next[i, u:,:])

                poly = PolynomialFeatures(2, interaction_only=True)
                mat_tmp = poly.fit_transform(np.vstack([Actions[i, :], States_current[i, :, range(p)]]).T)
                mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
                X.append(mat)
                y.append(States_next[i, :,:])

        X1 = np.vstack(np.array(X1))
        X1 = np.kron(np.eye(p,dtype=int),X1)
        y1 = np.vstack(np.array(y1)).T.reshape(-1, 1)
        X2 = np.vstack(np.array(X2))
        X2 = np.kron(np.eye(p,dtype=int),X2)
        y2 = np.vstack(np.array(y2)).T.reshape(-1, 1)
        X = np.vstack(np.array(X))
        X = np.kron(np.eye(p,dtype=int),X)
        y = np.vstack(np.array(y)).T.reshape(-1, 1)
        reg0 = LinearRegression(fit_intercept=False)
        res0 = reg0.fit(X, y)
        reg1 = LinearRegression(fit_intercept=False)
        res1=reg1.fit(X1, y1)
        reg2 = LinearRegression(fit_intercept=False)
        res2=reg2.fit(X2, y2)
        
        var0 = np.linalg.norm(y - res0.predict(X), ord=2) ** 2 / y.shape[0]
        # weigh the variance of each segment by their sample sizes: (# people) * (# time points) * (# dimension of States)
        var1 = (np.linalg.norm(y1 - res1.predict(X1), ord=2) ** 2
                + np.linalg.norm(y2 - res2.predict(X2), ord=2) ** 2) / y.shape[0]
        mean0 = res0.predict(X)
        mean1 = res1.predict(X1)
        mean2 = res2.predict(X2)
        # print(' var0', var0,' var1', var1)
        
        if var0 <  np.finfo(np.float64).tiny:
            var0 =  np.finfo(np.float64).tiny
        if var1 <  np.finfo(np.float64).tiny:
            var1 =  np.finfo(np.float64).tiny
        L0 = norm.pdf(y, loc = mean0, scale = np.sqrt(var0))
        # print('L0',L0)
        L1 = np.vstack([norm.pdf(y1, loc = mean1, scale = np.sqrt(var1)),norm.pdf(y2, loc = mean2, scale = np.sqrt(var1))])
        # print('L1',L1)
        L0[np.where(L0 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
        L1[np.where(L1 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
        logL0 = np.sum(np.log(L0))
        logL1 = np.sum(np.log(L1))
        cusum_tmp = -2*(logL0 - logL1)
        return cusum_tmp
    if nthread_B !=0:
        sample_stat = Parallel(n_jobs=nthread_B)(delayed(run_one)(seq) for seq in shift_index)
    else:
        sample_stat = [None for i in range(T-1)]
        for b in range(T-1):
            sample_stat[b] = run_one(shift_index[b])

    return sample_stat


#%% loops
# initilize with a clustering memembership
def clusteringNchangepoints(example, clustering, changepoint_detect, States,
                            Actions, N, T, p, epsilon, kappa_max, kappa_min, kappa_interval, K, cusum_forward,
                            cusum_backward, C1=1, C2=1/2, max_iter=30,
                            init_cluster_range=None, nthread=0, C=0, Kl_fun ='sqrtN',
                            ic_T_dynamic=0,
                            g_index_init = None,clustering_warm_start=1,
                            loss_path =1, threshold_type="maxcusum", 
                            init_cluster_method = 'kmeans',
                            distance_metric="correlation", linkage = "average", 
                            is_only_cp=0,
                            early_stopping=True):
    K_path = []
    if g_index_init is None:
        if init_cluster_range is None:
            # init_cluster_range = T - int(T/5) - 1
            init_cluster_range = int(T - kappa_min - 1)
        changepoints_0 =  np.tile(init_cluster_range, N)
        # print('changepoints_0',changepoints_0)
        if init_cluster_method == "kmeans":
            # print('kmeans')
            g_index, loss = clustering(States=States, N=N, T=T, K=K, changepoints=changepoints_0,g_index = None,
                                         example=example, Actions=Actions)
        elif init_cluster_method == 'hierarchy':
            g_index = ut.my_hierachy(States[:, init_cluster_range:, :], K,distance_metric, linkage)
    else:
        # print('default')
        g_index = g_index_init
        loss = 0
    if type(g_index) is int:
        g_index = np.array([g_index])
    print('g_index_0', g_index)
    
    K_path.append(len(set(g_index)))
    
    # if loss_path:
    loss_list = []
    loss_list.append(loss)
    changepoints_0 = np.zeros(N)
    iter_num=0
    min_loss = np.inf
    #%%
    for m in range(max_iter):
        print("======= m", m,'max_iter',max_iter, "=========", K_path)
        starttime = time.time()
        out=changepoint_detect(g_index = g_index,States=States, Actions=Actions, example=example, N=N,
                               T=T, kappa_max=kappa_max, kappa_min=kappa_min, kappa_interval=kappa_interval,
                               epsilon=epsilon, cusum_forward=cusum_forward,
                               cusum_backward=cusum_backward, C1=C1, C2=C2,
                               nthread=nthread)
        print('changepoint detect time', time.time() - starttime)
        
        changepoints = np.array(out[0])
        print('cp', np.unique(changepoints))
        
        if (m != 0 and np.prod(changepoints == changepoints_0) or m == max_iter -1):
            iter_num = m
            print('converge at iteration', m)
            break
        elif early_stopping:
            if np.abs(loss_list[-1] - min_loss) < 1e-5:
                loss = loss_list[-2]
                break
            elif loss_list[-1] < min_loss:
                min_loss = loss_list[-1]
        # else:
        if not is_only_cp:
            if clustering_warm_start:
                g_index, loss = clustering(States=States, Actions=Actions,example=example, g_index=g_index,
                                     N=N, T=T, K=K,changepoints=changepoints)
            else:
                g_index, loss = clustering(States=States, Actions=Actions, example=example, g_index=None,
                                     N=N, T=T, K=K, changepoints=changepoints)
        K_path.append(len(set(g_index)))
        loss_list.append(loss)
        if type(g_index) is int:
            g_index = np.array([g_index])
            
        changepoints_0 = changepoints.copy()
        iter_num = m
        sys.stdout.flush()
        #%%
    # out=changepoint_detect(g_index = g_index,States=States, Actions=Actions, example=example, N=N,
    #                        T=T, kappa_max=kappa_max, kappa_min=kappa_min, kappa_interval=kappa_interval,
    #                        epsilon=epsilon, cusum_forward=cusum_forward,
    #                        cusum_backward=cusum_backward, C1=C1, C2=C2,
    #                        nthread=nthread)
    # changepoints = np.array(out[0])
    
    ic = ut.IC(loss=loss, changepoints=changepoints,g_index= g_index, N=N, T=T, K=K_path[-1], C=C, Kl_fun=Kl_fun, ic_T_dynamic=ic_T_dynamic)
    try:
        result = namedtuple("result", ["iter_num", "g_index", "changepoints", "loss", "loss_list","IC",'K_path'])
        return result(iter_num, g_index, changepoints.flatten(), loss, loss_list, ic, K_path)
    except:
        result = namedtuple("result", ["iter_num", "g_index", "changepoints", "loss", "IC",'K_path'])
        return result(iter_num, g_index, changepoints.flatten(), loss, ic, K_path)



def changepointsNclustering(example, clustering, changepoint_detect, States, Actions,
                            N, T, p, epsilon, kappa_max, kappa_min, kappa_interval, K, cusum_forward,
                            cusum_backward, C1=1, C2=1/2,
                            max_iter=30, max_iter_gmr = 50, nthread=0, C=0, Kl_fun ='sqrtN',
                            ic_T_dynamic=0,
                            changepoints_init=None, g_index_init = None, clustering_warm_start=1,
                            loss_path = 1, threshold_type="chi2", changepoint_init_indi = 0,
                            is_only_cluster = 0, 
                            early_stopping=True):# is_only_cluster: for evalutation "only_cluster" type
    print('cploop nthread',nthread)
    # print("Kl_fun = ", Kl_fun)
    # if loss_path:
    loss_list = []
    if changepoint_init_indi == 1:
        g_index_0 = np.arange(0, N)
    else:
        g_index_0 = np.zeros([N])

    if changepoints_init is None:
        ## reserved for method start with individual change point detection
        # out = changepoint_detect(g_index = g_index_0,States=States, N=N, T=T,
        #                           kappa_max=kappa_max, kappa_min=kappa_min, kappa_interval=kappa_interval,
        #                           epsilon=epsilon,example=example,
        #                           Actions=Actions,
        #                           cusum_forward=cusum_forward, cusum_backward=cusum_backward,
        #                           C1=C1, C2=C2, nthread=nthread)
        # changepoints_0 = out[0]
        init_cluster_range = int(T - kappa_min - 1)
        changepoints_0 =  np.tile(init_cluster_range, N)
    else:
        changepoints_0 = changepoints_init
    g_index_0 = g_index_init
    iter_num = 0
    K_path=[]
    for m in range(max_iter):
        # print("======= m", m, "=========")
        if clustering_warm_start == 0:
            g_index,loss = clustering(States=States, Actions=Actions,example=example,
                                  N=N, T=T, K=K,changepoints=changepoints_0)
        else:
            g_index,loss = clustering(States=States, Actions=Actions,example=example, g_index=g_index_0,
                                  N=N, T=T, K=K,changepoints=changepoints_0)
        # print("g_index update",g_index)
        if type(g_index) is int:
            g_index = np.array([g_index])
        if is_only_cluster:
            changepoints = changepoints_init
        else:
            out=changepoint_detect(g_index = g_index,States=States, Actions=Actions,
                                   example=example,N=N, T=T,
                                   kappa_max=kappa_max, kappa_min=kappa_min,kappa_interval=kappa_interval,
                                   epsilon=epsilon,
                                   cusum_forward=cusum_forward, cusum_backward=cusum_backward,
                                   C1=C1, C2=C2,nthread=nthread)
            changepoints = np.array(out[0])
        # changepoint_list[:, [m+1]] = changepoints.reshape(N, 1)
        # g_index_list[:,[m]] = g_index.reshape(N,1)
        # if loss_path:
        loss_list.append(goodnessofClustering(States, N, T, changepoints, Actions, g_index))
        if m == 0:
            g_index_0 = -1*np.ones(N)
        if ((np.prod(changepoints == changepoints_0) and adjusted_rand_score(g_index_0.flatten(), g_index.flatten()) == 1)
            or m == max_iter-1):
            if m == max_iter-1 and (np.sum(changepoints != changepoints_0) or adjusted_rand_score(g_index_0.flatten(), g_index.flatten())<1): # not converge
                # print("K", str(K),": Not converge")
                # in case that changepoint detection does not converge
                # if loss_path == False:
                #     loss = goodnessofClustering(States, N, T, changepoints, Actions, g_index)
                # else:
                loss = loss_list[-1]

            ic = ut.IC(loss=loss, changepoints=changepoints, g_index=g_index, N=N, T=T, K=len(set(g_index)), C=C, Kl_fun=Kl_fun, ic_T_dynamic=ic_T_dynamic)
            # print("loss", loss, "ic", ic)
            iter_num = m
            break
        elif early_stopping and loss_path[-1] <= loss_path[-2]:
            loss = loss_path[-2]
            g_index = g_index_0
            changepoints = changepoints_0
            break
        else:
            changepoints_0 = changepoints.copy()
            g_index_0 = g_index.copy()
            K_path.append(len(set(g_index_0)))
        iter_num = m
    # if loss_path:
    # print(changepoints_0)
    try:
        result = namedtuple("result", ["iter_num", "g_index", "changepoints","loss", "loss_list","IC",'K_path'])
        return result(iter_num, g_index, changepoints, loss, loss_list, ic, K_path)
    except:
        result = namedtuple("result", ["iter_num", "g_index", "changepoints", "loss", "IC",'K_path'])
        return result(iter_num, g_index, changepoints, loss, ic, K_path)

#%% fit
def fit(States, Actions, example = "cdist", init = "changepoints", kappa_max = None,kappa_min = None, kappa_interval=None,epsilon=0.1, K=2,
        C1=1, C2=1/2,  alpha = 0.01, df=None, max_iter=1, init_cluster_range=None,
        max_iter_gmr = 100, seed = 1, nthread=3, C=2, Kl_fun = 'sqrtN',
        ic_T_dynamic=0,
        changepoints_init=None, g_index_init = None, clustering_warm_start=1,
        loss_path =0, threshold_type="maxcusum", nthread_B= None,  B=50000,
        init_cluster_method = 'kmeans',distance_metric="correlation", linkage = "average",
        changepoint_init_indi = 0,is_only_cluster = 0, is_only_cp=0,
        is_cp_parallel=0,is_tunek_wrap_parallel=0, early_stopping=True,save_path=None):
    '''
    :param example: "mean", "cdist"
    :param inti: initial estimator, "changepoints", detect changepoints for each trajectrory separately, "clusters", kemans
    :param loss_path: whether to calculate the loss of each iteration
    :param K: **an integer or a list of integers**. 
              when it is an iteger, run the algorithm with fixed K; 
              when it is a list, then tune K in each iteration
    :param threshold_type: "maxcusum": chi square approximation (default); "permutation".
    '''
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    # out_startTime = datetime.now()
    N = States.shape[0]
    T = States.shape[1]
    p = States.shape[2]
    #%%
    if kappa_max is None:
        kappa_max = T
    if kappa_min is None:
        kappa_min = max(25, int(2*epsilon*T))
    if kappa_interval is None:
        kappa_interval = 1
    if df is None:
        df = (2+2*p)*p

    # np.random.seed(seed)
    def changepoint_detect(g_index, States, N, T,  kappa_max, kappa_min,kappa_interval, epsilon, example=example, Actions=Actions,
                              cusum_forward=None, cusum_backward=None, C1=C1,
                              C2=C2, alpha = alpha, df = df,nthread=nthread,
                              threshold_type=threshold_type, nthread_B= None, B=B,is_cp_parallel=is_cp_parallel):
        # print('enter cp detect', example)
        if example == "mean":
            # print('cluster mean')
            return changemean_detect(g_index, States, N, T, kappa_max, kappa_min, kappa_interval,epsilon, Actions=None,
                                 cusum_forward=cusum_forward,
                                 cusum_backward=cusum_backward, C1=C1 , C2=C2, 
                                 alpha=None, df=None,nthread=nthread)
        elif example =="marginal":
            # print("marginal")
            return changemarginal_detect(g_index, States, N, T, kappa_max, kappa_min, kappa_interval,epsilon, Actions=Actions,
                                         cusum_forward=None, cusum_backward=None,
                                         C1=C1, C2=C2, alpha=alpha, df=df,nthread=nthread)
        elif example == "cdist":
            # print(1)
            return changedistribution_detect2(g_index, States, N, T, kappa_max, kappa_min,kappa_interval, epsilon, Actions=Actions,
                                         cusum_forward=None, cusum_backward=None,
                                         C1=None, C2=None, alpha=alpha,
                                         df=df,nthread=nthread,
                                         threshold_type=threshold_type, 
                                         nthread_B = nthread_B, B=B,
                                         is_cp_parallel=is_cp_parallel,
                                         save_path=save_path)

    def clustering(States, N, T, K, changepoints,example, Actions=None, g_index=g_index_init, 
                   max_iter_gmr = max_iter_gmr,C=C, is_tunek_wrap_parallel = is_tunek_wrap_parallel):
        '''
        Parameters
        ----------
        K : an integer or a list of integer. when it is an iteger, run the algorithm with fixed K; when it is a list, then tune K in each iteration
        '''
        if example == "mean" or example == "marginal":
            return clustering_mean(States, N, T, K, changepoints, Actions=None, g_index=None, max_iter_gmr=None)
        # elif example =="marginal":
        #     return clustering_marginal_dis(States, N, T, K, changepoints, Actions, g_index, max_iter_gmr)
        elif example == "cdist":
            if type(K) is int:
                return gmr(States, N, T, K, changepoints, Actions, g_index, max_iter_gmr)
            else: # tune K in each iteration  
                return tuneK_wrap(K, States, N, T, changepoints,Actions,g_index=None,
                                    max_iter_gmr=max_iter_gmr, C=C, is_tunek_wrap_parallel=is_tunek_wrap_parallel,Kl_fun=Kl_fun,
                                    ic_T_dynamic=ic_T_dynamic)

    if example == "mean":
        cusum_forward = np.cumsum(States, axis = 1)/(np.tile((range(1,T+1)), [N, 1]).reshape([N, T, p]))
        cusum_backward = np.flip(np.cumsum(np.flip(States, axis=1),axis = 1)/(np.tile((range(1,T+1)), [N, 1]).reshape([N, T, p])), axis = 1)
    else:
        cusum_forward = None
        cusum_backward = None
        #%%
    # fit
    print('init',init)
    if init == "changepoints":
        print('cp')
        result = changepointsNclustering(example, clustering, changepoint_detect, States, Actions,
                                         N, T, p, epsilon, kappa_max, kappa_min, kappa_interval, K, cusum_forward,
                                         cusum_backward, C1, C2,
                                         max_iter, max_iter_gmr, nthread, C, Kl_fun ,ic_T_dynamic,
                                         changepoints_init, g_index_init, clustering_warm_start,
                                         loss_path,
                                         threshold_type, changepoint_init_indi, 
                                         is_only_cluster, early_stopping)

    else:
        result = clusteringNchangepoints(example, clustering, changepoint_detect, States,
                                         Actions, N, T, p, epsilon, kappa_max, kappa_min, kappa_interval, K, cusum_forward,
                                         cusum_backward, C1, C2, max_iter,
                                         init_cluster_range, nthread, C, Kl_fun,ic_T_dynamic,
                                         g_index_init, clustering_warm_start,
                                         loss_path, threshold_type,
                                         init_cluster_method,
                                         distance_metric, linkage,is_only_cp, early_stopping)
    
    return result # , datetime.now() - out_startTime

def fit_tuneK(K_list, States, Actions, example = "cdist", init = "changepoints", kappa_max = None, kappa_min=None,kappa_interval=None,epsilon=0.1,
        C1=1, C2=1/2, alpha = 0.01, df=None, max_iter=1, init_cluster_range=None,
        max_iter_gmr = 50, seed = 1, nthread=0, C=2, Kl_fun = 'sqrtN', ic_T_dynamic=0,
        changepoints_init=None,
        g_index_init_list = None, clustering_warm_start=1, loss_path =0,
        threshold_type="maxcusum", nthread_B= None, B=50000, init_cluster_method = 'kmeans',
        distance_metric="correlation", linkage = "average", changepoint_init_indi = 0,
        is_only_cluster = 0, is_only_cp=0, is_cp_parallel=0, is_tunek_wrap_parallel=0,
        only_best=1, is_tune_parallel=0, early_stopping=1,save_path=None):
    '''
    Tuning the best K for clustering initialization from a list. When running the algorithm, K is fixed.
    #param: K_list: list object
    #param: C: constant for information critirion
    '''
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    out_startTime=datetime.now()
    IC_max = 0
    K_max = None
    res = {}
    if only_best!=0:
        IC_model = []
        loss_model = []
    best_model =None
    if g_index_init_list is None:
        g_index_init_list = [None]*len(K_list)
    #%%
    def run_K(K):
        out = fit(States, Actions, example, init, kappa_max,kappa_min, kappa_interval,epsilon, K,
                C1, C2, alpha, df, max_iter, init_cluster_range,
                max_iter_gmr, seed, nthread, C, Kl_fun, ic_T_dynamic,
                changepoints_init, g_index_init_list[K_list.index(K)],
                clustering_warm_start, loss_path,threshold_type, nthread_B, B,
                init_cluster_method,distance_metric, linkage, changepoint_init_indi, 
                is_only_cluster, is_only_cp, is_cp_parallel, is_tunek_wrap_parallel,
                early_stopping,save_path=save_path)
        return out
    if is_tune_parallel ==0:
        for K in K_list:
            out = run_K(K)
            res[K] = out
            print('K', K, ', loss:', out.loss, ', ic:',out.IC)
            if only_best!=0:
                IC_model.append(out.IC)
                loss_model.append(out.loss)
            if K == K_list[0]:
                IC_max = out.IC
                K_max = K
                best_model = out
            elif out.IC > IC_max:
                IC_max = out.IC
                K_max = K
                best_model = out
    else:
        tune_thread = len(K_list)
        IC_model = Parallel(n_jobs=tune_thread)(delayed(run_K)(K) for K in K_list)
        for K in K_list:
            print('K', K, ', los:', IC_model[K_list.index(K)].loss, ', ic:', IC_model[K_list.index(K)].IC)
            # ic = ut.IC(IC_model[K_list.index(K)].loss, IC_model[K_list.index(K)].changepoints, IC_model[K_list.index(K)].g_index, States.shape[0], States.shape[1], K, C)
            if only_best!=0:
                loss_model.append(IC_model[K_list.index(K)].loss)
            if K == K_list[0]:
                IC_max = IC_model[K_list.index(K)].IC
                K_max = K
                best_model = IC_model[K_list.index(K)]
            elif IC_model[K_list.index(K)].IC > IC_max:
                IC_max = IC_model[K_list.index(K)].IC
                K_max = K
                best_model = IC_model[K_list.index(K)]
    #%%
    print('bestK:', K_max)
    if only_best:
        tunningres = namedtuple("tunningres", ["best_K", "IC", "best_model"])
        return tunningres(K_max, IC_max, best_model) #, datetime.now() - out_startTime
    else:
        tunningres = namedtuple("tunningres", ["K", "IC", "best_model", "models",
                                               "IC_model", "loss_model"])
        return tunningres(K_max, IC_max, res[K_max], res, IC_model,
                          loss_model)  