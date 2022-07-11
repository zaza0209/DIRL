"""
Detect change points and clusters in mean.
"""
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.cluster import adjusted_rand_score
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.linalg import block_diag
from scipy.stats import chi2, norm
from joblib import Parallel, delayed
import simu.utilities as ut

#%%
def goodnessofClustering(States, N, T, K, changepoints,Actions, g_index):
    p = States.shape[2]
    g_index = g_index.astype(int)
    changepoints = changepoints.astype(int)

    mat_list = [[] for i in range(K)]
    y = [[] for i in range(K)]
    Xi = [[] for i in range(N)]
    yi = [[] for i in range(N)]
    weights = []
    for i in range(int(N)):
        g = g_index.item(i)
        # construct the design matrix
        mat = []
        poly = PolynomialFeatures(2, interaction_only=True)
        mat_tmp = poly.fit_transform(np.vstack([Actions[i, changepoints.item(i):], States[i, changepoints.item(i):-1, range(p)]]).T)
        mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
        mat_list[g].extend(mat.tolist())
        y[g].extend(States[i, changepoints.item(i)+1:,:].tolist())
        Xi[i] = mat
        yi[i] = States[i, changepoints.item(i)+1:,:]
        weights.extend([1/mat.shape[0] for r in range(p*mat.shape[0])])
    for g in range(K):
        # print(g)
        y[g] = np.array(y[g]).T.reshape(-1,1)
        mat_list[g] = np.vstack(np.array(mat_list[g]))
        mat_list[g] = np.kron(np.eye(p,dtype=int),mat_list[g])

    X = block_diag(*mat_list)
    y = np.vstack(y)

    reg = LinearRegression(fit_intercept=False) # coef: [beta1, beta2, ..., betaK] for K clusters
    res=reg.fit(X, y, sample_weight=weights)
    loss =  0
    for i in range(N):
        Xistack = np.kron(np.eye(K*p,dtype=int),Xi[i])
        yhat = res.predict(Xistack)
        err = []
        t = Xi[i].shape[0]
        for k in range(K):
            if g_index[i] == k:
                for j in range(p):
                    err.append(np.linalg.norm((yi[i][:,j].reshape([t,1]) - yhat[(j + k * p)*t :(j + k * p)*t + t]), ord=2) ** 2)
        err = np.sum(np.array(err))
        loss = loss - 1 * err/t
    loss = loss * np.mean(T - changepoints - 1)
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
        # print('m',m)
        for i in range(N):
            err = []
            for k in range(K):
                err.append(np.linalg.norm((yi[i] - mean[k]), ord=2)**2/var[k])
            g_index_new[i] = err.index(min(err))
        # print('g_index_new',g_index_new)
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

def gmr(States, N, T, K, changepoints,Actions, g_index=None, max_iter_gmr = 50):
    '''
    mixutre regression model for normal conditional distribution
    Parameters
    ----------
    g_index : the initial group index, optional
        DESCRIPTION. The default is None.
    -------
    '''
    p = States.shape[2]
    if g_index is None:
        # print(N)
        # print(K)
        g_index = np.random.choice(range(K), size = N)
    # g_index = np.append(np.zeros(int(N/2)), np.ones(int(N/2)))

    g_index = g_index.astype(int)
    # print('gmr: beginning g_index', g_index)
    changepoints = changepoints.astype(int)

    mat_list = [[] for i in range(K)]
    y = [[] for i in range(K)]
    Xi = [[] for i in range(N)]
    yi = [[] for i in range(N)]
    weights = []
    for i in range(int(N)):
        g = g_index.item(i)
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
    for m in range(max_iter_gmr):
        # print("---gmr m,",m,"---")
        # print(res.coef_[:6])
        err_all = np.zeros([N, K])
        for i in range(N):
            Xistack = np.kron(np.eye(K*p,dtype=int),Xi[i])
            yhat = res.predict(Xistack)
            err = [[] for i in range(K)]
            t = Xi[i].shape[0]
            for k in range(K):
                for j in range(p):
                    err[k].append(np.linalg.norm((yi[i][:,j].reshape([t,1]) - yhat[(j + k * p)*t :(j + k * p)*t + t]), ord=2) ** 2)

            err = np.sum(np.array(err), axis=1)
            err_all[i, :] = err.reshape(K)
            # print('err_all[',i,', :]', err_all[i, :])
            g_index_new[i] = np.where(err==min(err))[0][0]
            loss = loss - 1 * min(err)/t
        # plt.plot(g_index_new)
        # keep the cluster size unchanged
        if np.unique(g_index_new).shape[0] < K:
            # print("cluster size changed",( np.setdiff1d(np.array(range(K)),np.unique(g_index))))
            # print('err_all', err_all)
            i_tmp = -1*np.ones(len((np.setdiff1d(np.array(range(K)),np.unique(g_index_new)).tolist())))
            z=0
            for k in (np.setdiff1d(np.array(range(K)),np.unique(g_index_new)).tolist()):
                # print('i_tmp[',z,']',i_tmp[z],np.where(err_all[:,k] == min(err_all[:,k])))
                smallest = 0
                assign_ind = np.where(err_all[:,k] == min(err_all[:,k]))[0]
                # print("assign_ind", assign_ind)
                while (assign_ind in i_tmp) or (np.sum(g_index_new == g_index_new[assign_ind])==1):
                    smallest = smallest+1
                    # print(smallest)
                    assign_ind = np.where(err_all[:,k] == np.partition(err_all[:,k], smallest)[smallest])[0]
                    # print("assign_ind", assign_ind)
                loss = loss + err_all[assign_ind,g_index_new[assign_ind]] - err_all[assign_ind,k]
                g_index_new[assign_ind] = k
                i_tmp[z]=assign_ind
                z = z+1
                # print('k',k,'np.where(err_all[:,k] == min(err_all[:,k]))',np.where(err_all[:,k] == min(err_all[:,k])))
                # g_index[np.where(err_all[:,k] == min(err_all[:,k]))] = k
            # print("g_index_new", g_index)
        loss = loss * np.mean(T - changepoints - 1)
        # plt.figure()
        # plt.plot(g_index_new)
        # plt.show()
        # print("g_index_new", g_index_new)
        # print("np.prod(g_index == g_index_new)",np.prod(g_index == g_index_new))
        if np.prod(g_index == g_index_new):
            break
        elif not np.prod(g_index == g_index_new) or m == max_iter_gmr - 1:
            loss = 0
            g_index = g_index_new.copy()
            mat_list = [[] for i in range(K)]
            y = [[] for i in range(K)]
            weights = []
            for i in range(int(N)):
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
    return g_index, loss


#%% changepoint detection functions
def changemean_detect(g_index, States, N, T, kappa_max,kappa_min, epsilon,Actions=None, cusum_forward=None,
                      cusum_backward=None, C1=None, C2=None, C3=None, alpha = 0.05, df=5, nthread=0):
    '''
    detect change in means
    need adaptations for kappa
    '''
    K = len(set(g_index))
    # K = kmeans.cluster_centers_.shape[0]
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

def changemarginal_detect(g_index, States, N, T, kappa_max, kappa_min, epsilon, Actions=None,
                              cusum_forward=None, cusum_backward=None, C1=None,
                              C2=None, C3=None, alpha = 0.05, df = 5, nthread=0):
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
            if nthread !=0: # parallel
                res = Parallel(n_jobs=nthread)(delayed(run_one)(u) for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1))
                tauk[k] = int(T-epsilon*T)-1- res.index(np.max(res))
                maxcusum= np.max(res)
            else: # do not parallel
                cusum_list = []
                for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1):
                    cusum_tmp = run_one(u)
                    # print('k',k,'u',u)
                    # y1 = []
                    # y2 = []
                    # y = []
                    # for i in range(int(N)):
                    #     if g_index[i] == k:
                    #         y1.append(States[i, :u+1,:])
                    #         y2.append(States[i, u+1:,:])
                    #         y.append(States[i, :,:])
                    # y1 = np.vstack(np.array(y1))
                    # y2 = np.vstack(np.array(y2))
                    # y = np.vstack(np.array(y))
                    # var0 = np.var(y)
                    # var1 = (np.var(y1) * np.size(y1)+np.var(y2)*np.size(y2))/(T*np.sum(g_index == k))
                    # mean0 = np.mean(y)
                    # mean1 = np.mean(y1)
                    # mean2 = np.mean(y2)
                    # L0 = norm.pdf(y, loc = mean0, scale = np.sqrt(var0))
                    # L1 = np.vstack([norm.pdf(y1, loc = mean1, scale = np.sqrt(var1)),norm.pdf(y2, loc = mean2, scale = np.sqrt(var1))])
                    # L0[np.where(L0 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
                    # L1[np.where(L1 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
                    # logL0 = np.sum(np.log(L0))
                    # logL1 = np.sum(np.log(L1))
                    # cusum_tmp =-1*(logL0 - logL1)/(kappa * np.sum(g_index == k)) #-2*(logL0 - logL1)
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
    return [changepoints, maxcusum_list, tauk]

def changedistribution_detect2(g_index, States, N, T,kappa_max,  kappa_min,  epsilon, Actions=None,
                              cusum_forward=None, cusum_backward=None, C1=None,
                              C2=None,C3=2, alpha = 0.05, df = 4, nthread=0):
    '''
    detect change in conditional distribution, no variance estimation
    need further adaptations for changing kappa
    '''
    p = States.shape[2]
    K = len(set(g_index.tolist()))
    tauk = np.zeros(K)
    changepoints = np.zeros(N).reshape(-1,1)
    maxcusum_list = np.zeros(K)
    g_index = np.array(g_index).astype(int)
    def run_one(u):
       X1 = []
       y1 = []
       X2 = []
       y2 = []
       X = []
       y = []
       for i in range(int(N)):
            if g_index[i] == k:
               poly = PolynomialFeatures(2, interaction_only=True)
               mat_tmp = poly.fit_transform(np.vstack([Actions[i, :u], States[i, :u, range(p)]]).T)
               mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
               X1.append(mat)
               y1.append(States[i, 1:u+1,:])

               poly = PolynomialFeatures(2, interaction_only=True)
               mat_tmp = poly.fit_transform(np.vstack([Actions[i, u:], States[i, u:-1, range(p)]]).T)
               mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
               X2.append(mat)
               y2.append(States[i, u+1:,:])

               poly = PolynomialFeatures(2, interaction_only=True)
               mat_tmp = poly.fit_transform(np.vstack([Actions[i, :], States[i, :-1, range(p)]]).T)
               mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
               X.append(mat)
               y.append(States[i, 1:,:])
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
       # var0 = np.linalg.norm(y - res0.predict(X), ord=2)**2/N
       # var1 = (np.log(np.linalg.norm(y1 - res1.predict(X1), ord=2)**2)
       #      +np.log(np.linalg.norm(y2 - res2.predict(X2),ord=2)**2))/N
       mean0 = res0.predict(X)
       mean1 = res1.predict(X1)
       mean2 = res2.predict(X2)
       # L0 = norm.pdf(y, loc = mean0, scale = np.sqrt(var0))
       # L1 = np.vstack([norm.pdf(y1, loc = mean1, scale = np.sqrt(var1)),norm.pdf(y2, loc = mean2, scale = np.sqrt(var1))])
       # L0[np.where(L0 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
       # L1[np.where(L1 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
       H0 = ((y - mean0).T).dot(y-mean0)
       H1 = ((y1 - mean1).T).dot(y1 - mean1) + (y2 - mean2).T.dot(y2 - mean2)
       # logL0 = np.sum(np.log(L0))
       # logL1 = np.sum(np.log(L1))
       cusum_tmp = H0-H1
       return cusum_tmp

    for k in range(K):
        # print("k",k)
        for kappa in range(kappa_min, kappa_max):
            maxcusum = 0
            cusum_list = []
            if nthread !=0: # parallel
                res = Parallel(n_jobs=nthread)(delayed(run_one)(u) for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1))
                tauk[k] = int(T-epsilon*T)-1- res.index(np.max(res))
                maxcusum= np.max(res)
            else: # do not parallel
                # print("no parallel")
                # print(g_index)
                for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1):
                    # print("u",u, "K",K)
                    cusum_tmp = run_one(u)
                    cusum_list.append(cusum_tmp[0])
                    # cusum_list.index(max(cusum_list))
                    # plt.plot(cusum_list)
                    if u == int(T-epsilon*T)-1:
                        maxcusum = cusum_tmp
                        maxcusum_list[k] = maxcusum
                        tauk[k] = u
                    elif maxcusum <= cusum_tmp:
                        # print('** maxcusum',maxcusum)
                        maxcusum = cusum_tmp
                        maxcusum_list[k] = maxcusum
                        tauk[k] = u
                        # plt.plot(cusum_list)
            # print('np.sum(g_index==k)',g_index==k,np.sum(g_index==k))
            # print(g_index)
            # print(maxcusum)
            # print('chi2.ppf(',1-alpha,', df) * np.log10(T * np.sum(g_index==k))**2')
            # print('df',df,'np.sum(g_index==k)')
            # print('np.sum(g_index==k)',np.sum(g_index==k))
    
            # print('np.log10(T * np.sum(g_index==k))',np.log10(T * np.sum(g_index==k)))
            # print('chi2.ppf(1-alpha, df)',chi2.ppf(1-alpha, df))
            # print(chi2.ppf(1-alpha, df) * np.log10(T * np.sum(g_index==k))**2)
            if maxcusum < chi2.ppf(1-alpha, df) * np.log10(kappa * np.sum(g_index==k))**C3:
                tauk[k] = 0
            changepoints[(g_index == k), :] = tauk[k]
    changepoints = changepoints.astype(int)
    return [changepoints, maxcusum_list, tauk]

def permutation_test(States_ori, Actions_ori, g_index, k, u, B, nthread_B=1):
    N = States_ori.shape[0]
    T = States_ori.shape[1] 
    p = States_ori.shape[2]
    def run_one(b):
        # np.random.seed(None)
        # States_current = States_ori[:, :-1, :]
        # States_next = States_ori[:, 1:, :]
        seq = np.random.permutation(range(T-1))
        States_current = States_ori[:,seq,:]
        States_next = States_ori[:, seq+1, :]
        Actions = Actions_ori[:,seq]
        X1 = []
        y1 = []
        X2 = []
        y2 = []
        X = []
        y = []
        N_k = sum(g_index == k)
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

        var0 = np.linalg.norm(y - res0.predict(X), ord=2)**2/(N_k*p*(T-1))
        var1 = (np.linalg.norm(y1 - res1.predict(X1), ord=2)**2 / u
                + np.linalg.norm(y2 - res2.predict(X2),ord=2)**2 / (T-u-1))/(N_k*p)
        mean0 = res0.predict(X)
        mean1 = res1.predict(X1)
        mean2 = res2.predict(X2)

        L0 = norm.pdf(y, loc = mean0, scale = np.sqrt(var0))
        L1 = np.vstack([norm.pdf(y1, loc = mean1, scale = np.sqrt(var1)),norm.pdf(y2, loc = mean2, scale = np.sqrt(var1))])
        L0[np.where(L0 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
        L1[np.where(L1 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
        logL0 = np.sum(np.log(L0))
        logL1 = np.sum(np.log(L1))
        cusum_tmp = -2*(logL0 - logL1)
        return cusum_tmp

    if nthread_B !=0:
        sample_stat = Parallel(n_jobs=nthread_B)(delayed(run_one)(b) for b in range(B))
    else:
        sample_stat = [None for i in range(B)]
        for b in range(B):
            sample_stat[b] = run_one(b)
    return sample_stat

def changedistribution_detect(g_index, States, N, T, kappa_max, kappa_min, epsilon, Actions=None,
                              cusum_forward=None, cusum_backward=None, C1=None,
                              C2=None,C3=2, alpha = 0.05, df = 5,
                              nthread=0, threshold_type="permutate", B = 100, nthread_B= None):
    '''
    detect change in conditional distribution
    '''
    K = len(set(g_index))
    # print(g_index)
    p = States.shape[2]
    tauk = np.zeros(K)
    changepoints = np.zeros(N).reshape(-1,1)
    maxcusum_list = np.zeros(K)
    def run_one(u, kappa):
       X1 = []
       y1 = []
       X2 = []
       y2 = []
       X = []
       y = []
       N_k = sum(g_index == k)
       # print('g_index',g_index)
       for i in range(int(N)):
           if g_index[i] == k:
               # the first intercept of A and S
               mat = []
               poly = PolynomialFeatures(2, interaction_only=True)
               mat_tmp = poly.fit_transform(np.vstack([Actions[i, int(T-kappa):u], States[i, int(T-kappa):u, range(p)]]).T)
               mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
               X1.append(mat)
               y1.append(States[i, int(T-kappa+1):u+1,:])
               
               # the 2nd intercept of A and S
               mat = []
               poly = PolynomialFeatures(2, interaction_only=True)
               mat_tmp = poly.fit_transform(np.vstack([Actions[i, u:], States[i, u:-1, range(p)]]).T)
               mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
               X2.append(mat)
               y2.append(States[i, u+1:,:])
              
               # the whole period of A and S
               mat = []
               poly = PolynomialFeatures(2, interaction_only=True)
               mat_tmp = poly.fit_transform(np.vstack([Actions[i, int(T-kappa):], States[i, int(T-kappa):-1, range(p)]]).T)
               mat = np.delete(mat_tmp, np.s_[2 + p*2:mat_tmp.shape[1]], 1)
               X.append(mat)
               y.append(States[i, int(T-kappa+1):,:])

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

       var0 = np.linalg.norm(y - res0.predict(X), ord=2)**2/(N_k*p*(T-int(T-kappa+1))) #
       var1 = (np.linalg.norm(y1 - res1.predict(X1), ord=2)**2 / (u-int(T-kappa))
             + np.linalg.norm(y2 - res2.predict(X2),ord=2)**2 / (T-u-1))/(N_k * p) #
       mean0 = res0.predict(X)
       mean1 = res1.predict(X1)
       mean2 = res2.predict(X2)

       L0 = norm.pdf(y, loc = mean0, scale = np.sqrt(var0))
       L1 = np.vstack([norm.pdf(y1, loc = mean1, scale = np.sqrt(var1)),norm.pdf(y2, loc = mean2, scale = np.sqrt(var1))])
       L0[np.where(L0 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
       L1[np.where(L1 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
       logL0 = np.sum(np.log(L0))
       logL1 = np.sum(np.log(L1))
       cusum_tmp = -2*(logL0 - logL1)
       # print(cusum_tmp)
       # print(np.linalg.norm(mean1-y1,ord=2), np.linalg.norm(mean2 - y2, ord=2), np.linalg.norm(mean0-y,ord=2))
       # print(res1.coef_)
       # print(res2.coef_)
       return cusum_tmp

    for k in range(K):
        for kappa in range(kappa_min, kappa_max):
            is_cp_found=1
            maxcusum = 0
            cusum_list = []
            if nthread !=0: # parallel
                res = Parallel(n_jobs=nthread)(delayed(run_one)(u, kappa) for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1))
                tauk[k] = int(T-epsilon*T)-1- res.index(np.max(res))
                maxcusum= np.max(res)
            else: # do not parallel
                for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1):
                    # print('u',u)
                    cusum_tmp = run_one(u, kappa)
                    cusum_list.append(cusum_tmp)
                    # cusum_list.index(max(cusum_list))
                    # plt.plot(cusum_list)
                    if u == int(T-epsilon*T)-1:
                        maxcusum = cusum_tmp
                        maxcusum_list[k] = maxcusum
                        tauk[k] = u
                    elif maxcusum <= cusum_tmp:
                        # print('** maxcusum',maxcusum)
                        maxcusum = cusum_tmp
                        maxcusum_list[k] = maxcusum
                        tauk[k] = u
            if threshold_type == "Chi2":
                threshold = chi2.ppf(1-alpha, df)
            elif threshold_type == "permutate":
                if nthread_B is None:
                    nthread_B = nthread
                sample_stat = permutation_test(States[:, int(T-kappa):, ], Actions[:, int(T-kappa):], g_index, k, u-int(T-kappa), B, nthread_B)
                threshold = np.percentile(sample_stat, (1- alpha)*100)
            if maxcusum < threshold:
                tauk[k] = 0
                is_cp_found=0
            changepoints[(g_index == k), :] = tauk[k]
            if is_cp_found:
                break
    changepoints = changepoints.astype(int)
    return [changepoints, maxcusum_list, tauk]

#%% loops
def clusteringNchangepoints(example, clustering, changepoint_detect, States,
                            Actions, N, T, p, epsilon, kappa_max, kappa_min, K, cusum_forward,
                            cusum_backward, C1=1, C2=1/2,C3=2, max_iter=30,
                            init_cluster_range=None, nthread=0, C=1,
                            g_index_init = None,clustering_warm_start=1,
                            loss_path =0, threshold_type="permutate", B = 10, 
                            nthread_B= None, init_cluster_method = 'gmr',distance_metric="correlation", linkage = "average"):
    if g_index_init is None:
        if init_cluster_range is None:
            init_cluster_range = int(T/4) - 1
        changepoints_0 =  np.tile(init_cluster_range, N)
        if init_cluster_method == "gmr":
            g_index_0, loss = clustering(States=States,Actions=Actions,example=example, N=N, T=T, K=K,
                                   changepoints=changepoints_0)
        elif init_cluster_method == 'hierarchy':
            g_index_0 = ut.my_hierachy(States[:, init_cluster_range:, :], K,distance_metric, linkage)
        elif init_cluster_method == 'kmeans':
            km = TimeSeriesKMeans(n_clusters=K, metric=distance_metric,
                       random_state=0).fit(States[:, init_cluster_range:, :])
            g_index_0 = km.labels_
    else:
        g_index_0 = g_index_init
    print('g_index_0', g_index_0)
    changepoint_list = np.zeros([N, max_iter+1])
    g_index_list = np.zeros([N, max_iter+1])
    if loss_path:
        loss_list = np.zeros(max_iter+1)
    changepoints_0 = np.zeros(N)
    g_index_list[:,0] = g_index_0
    # changepoint_list[:, [0]] = changepoints_0.reshape(N, 1)
    iter_num=0
    for m in range(max_iter):
        # print("======= m", m, "=========")
        out=changepoint_detect(g_index = g_index_0,States=States, Actions=Actions,example=example, N=N,
                               T=T, kappa_max=kappa_max, kappa_min=kappa_min,
                               epsilon=epsilon, cusum_forward=cusum_forward,
                               cusum_backward=cusum_backward, C1=C1, C2=C2,nthread=nthread)
        changepoints = np.array(out[0])
        if loss_path:
            loss_list[m] = goodnessofClustering(States, N, T, K, changepoints, Actions, g_index_0)
        # print('cp',changepoints)
        # print('g ori', g_index_0,'changepoints_0',changepoints_0)
        if clustering_warm_start:
            g_index, loss = clustering(States=States, Actions=Actions,example=example, g_index=g_index_0,
                                 N=N, T=T, K=K,changepoints=changepoints)
        else:
            g_index, loss = clustering(States=States, Actions=Actions,example=example, g_index=None,
                                 N=N, T=T, K=K,changepoints=changepoints)
        changepoint_list[:, [m]] = changepoints.reshape(N, 1)
        g_index_list[:,[m+1]] = g_index.reshape(N, 1)
        # print('g',g_index)
        if (m != 0 and(np.prod(changepoints == changepoints_0) and adjusted_rand_score(g_index_0.flatten(), g_index.flatten())) or m == max_iter -1):
            iter_num = m
            break
        else:
            changepoints_0 = changepoints.copy()
            g_index_0 = g_index.copy()
        iter_num = m
    out=changepoint_detect(g_index = g_index,States=States, Actions=Actions,example=example, N=N, T=T, 
                           kappa_max=kappa_max, kappa_min=kappa_min,
                           epsilon=epsilon, cusum_forward=cusum_forward,
                           cusum_backward=cusum_backward, C1=C1, C2=C2,nthread=nthread)
    changepoints = np.array(out[0])
    # print('cp',changepoints)
    loss = goodnessofClustering(States, N, T, K, changepoints, Actions, g_index)
    if loss_path:
        loss_list[m + 1] = goodnessofClustering(States, N, T, K, changepoints, Actions, g_index_0)
    ic = ut.IC(loss, changepoints, g_index, N, T, K, C)
    changepoint_list= np.hstack([changepoint_list[:, :m], changepoints])
    g_index_list = g_index_list[:, :(m+1)]
    # print('m',m)
    if loss_path:
        print('loss_path',loss_path,loss_list)
        loss_list = loss_list[:iter_num]
    try:
        result = namedtuple("result", ["iter_num", "g_index", "changepoints",
                                       "changepoint_eachiter","g_index_eachiter", "loss", "loss_list","IC"])
        return result(iter_num, g_index, changepoints, changepoint_list, g_index_list, loss, loss_list, ic)
    except:
        result = namedtuple("result", ["iter_num", "g_index", "changepoints",
                                       "changepoint_eachiter","g_index_eachiter", "loss", "IC"])
        return result(iter_num, g_index, changepoints, changepoint_list, g_index_list, loss, ic)

def changepointsNclustering(example, clustering, changepoint_detect, States,Actions,
                            N, T, p, epsilon, kappa_max, kappa_min, K, cusum_forward,
                            cusum_backward, C1=1, C2=1/2, C3=2,
                            max_iter=30, max_iter_gmr = 50, nthread=0, C=1,
                            changepoints_init=None,clustering_warm_start=1,
                            loss_path = 0,threshold_type="permutate", B = 10, nthread_B= None):
    changepoint_list = np.zeros([N, max_iter+1])
    g_index_list = np.zeros([N, max_iter+1])
    if loss_path:
        loss_list = np.zeros(max_iter+1)
    if changepoints_init is None:
        g_index = np.arange(0, N)
        out = changepoint_detect(g_index = g_index,States=States, N=N, T=T,
                                 kappa_max=kappa_max, kappa_min=kappa_min,
                                 epsilon=epsilon,example=example, 
                                 Actions=Actions,
                                 cusum_forward=cusum_forward, cusum_backward=cusum_backward,
                                 C1=C1, C2=C2, nthread=nthread)
        changepoints_0 = out[0]
    else:
        changepoints_0 = changepoints_init
    # print(changepoints_0)
    g_index_0 = None
    changepoint_list[:, [0]] = changepoints_0.reshape(N, 1)
    iter_num = 0
    # g_index_0 = g_index
    for m in range(max_iter):
        # print("======= m", m, "=========")
        if clustering_warm_start == 0:
            g_index,loss = clustering(States=States, Actions=Actions,example=example,
                                  N=N, T=T, K=K,changepoints=changepoints_0)
        else:
            g_index,loss = clustering(States=States, Actions=Actions,example=example, g_index=g_index_0,
                                  N=N, T=T, K=K,changepoints=changepoints_0)
        # print("g_index update",g_index)
        out=changepoint_detect(g_index = g_index,States=States, Actions=Actions, 
                               example=example,N=N, T=T, 
                               kappa_max=kappa_max, kappa_min=kappa_min,
                               epsilon=epsilon,
                                 cusum_forward=cusum_forward, cusum_backward=cusum_backward,
                                 C1=C1, C2=C2,nthread=nthread)
        changepoints = np.array(out[0])
        changepoint_list[:, [m+1]] = changepoints.reshape(N, 1)
        g_index_list[:,[m]] = g_index.reshape(N,1)
        if loss_path:
            loss_list[m] = goodnessofClustering(States, N, T, K, changepoints, Actions, g_index)
        if m == 0:
            g_index_0 = -1*np.ones(N)
        if ((np.prod(changepoints == changepoints_0) and adjusted_rand_score(g_index_0.flatten(), g_index.flatten()) == 1)
            or m == max_iter-1):
            if m == max_iter-1 and (np.sum(changepoints != changepoints_0) or adjusted_rand_score(g_index_0.flatten(), g_index.flatten())<1): # not converge
                # print("K", str(K),": Not converge")
                # in case that changepoint detection does not converge
                if loss_path == False:
                    loss = goodnessofClustering(States, N, T, K, changepoints, Actions, g_index)
                else:
                    loss = loss_path[m]
            ic = ut.IC(loss, changepoints, g_index, N, T, K, C)
            # print("loss", loss, "ic", ic)
            iter_num = m
            break
        else:
            changepoints_0 = changepoints.copy()
            g_index_0 = g_index.copy()
        iter_num = m
    changepoint_list = changepoint_list[:, :(iter_num+1)]
    g_index_list = g_index_list[:, :iter_num]
    if loss_path:
        loss_list = loss_list[:iter_num]
    # print(changepoints_0)
    try:
        result = namedtuple("result", ["iter_num", "g_index", "changepoints",
                                       "changepoint_eachiter","g_index_eachiter", "loss", "loss_list","IC"])
        return result(iter_num, g_index, changepoints, changepoint_list, g_index_list, loss, loss_list, ic)
    except:
        result = namedtuple("result", ["iter_num", "g_index", "changepoints",
                                       "changepoint_eachiter","g_index_eachiter", "loss", "IC"])
        return result(iter_num, g_index, changepoints, changepoint_list, g_index_list, loss, ic)

#%% fit
def fit(States, Actions, example = "mean", init = "changepoints", kappa_max = None,kappa_min = None, epsilon=0.1, K=2,
        C1=1, C2=1/2, C3=2, alpha = 0.05, df=None, max_iter=5, init_cluster_range=None,
        max_iter_gmr = 50, seed = 1, nthread=0, C=1,
        changepoints_init=None,g_index_init = None,clustering_warm_start=1,
        loss_path =0, threshold_type="Chi2", B = 10, nthread_B= None,init_cluster_method = 'kmeans',distance_metric="correlation", linkage = "average"):
    '''
    :param example: "mean", "conditional dis"
    :param inti: initial estimator, "changepoints", detect changepoints for each trajectrory separately, "clusters", kemans
    :param loss_path: whether to calculate the loss of each iteration
    '''
    N = States.shape[0]
    # print(N)
    T = States.shape[1]
    p = States.shape[2]
    #%%
    if kappa_max is None:
        kappa_max = T
    if kappa_min is None:
        kappa_min = max(25, int(2*epsilon*T))
    if df is None:
        df = (2+2*p)*p

    np.random.seed(seed)
    def changepoint_detect(g_index, States, N, T,  kappa_max, kappa_min, epsilon, example=example, Actions=Actions,
                              cusum_forward=None, cusum_backward=None, C1=C1,
                              C2=C2, C3=C3,alpha = alpha, df = df,nthread=nthread,
                              threshold_type="Chi2", B = 10, nthread_B= None):
        if example == "mean":
            # print('cluster mean')
            return changemean_detect(g_index, States, N, T, kappa_max, kappa_min, epsilon, Actions=None,
                                 cusum_forward=cusum_forward,
                                 cusum_backward=cusum_backward, C1=C1 , C2=C2, C3=C3,
                                 alpha=None, df=None,nthread=nthread)
        elif example =="marginal":
            # print("marginal")
            return changemarginal_detect(g_index, States, N, T, kappa_max, kappa_min, epsilon, Actions=Actions,
                                         cusum_forward=None, cusum_backward=None,
                                         C1=C1, C2=C2,C3=C3, alpha=alpha, df=df,nthread=nthread)
        elif example == "cdist":
            # print(1)
            return changedistribution_detect(g_index, States, N, T, kappa_max, kappa_min, epsilon, Actions=Actions,
                                         cusum_forward=None, cusum_backward=None,
                                         C1=None, C2=None,C3= C3, alpha=alpha,
                                         df=df,nthread=nthread,
                                         threshold_type=threshold_type, B = B, nthread_B = nthread_B)

    def clustering(States, N, T, K, changepoints,example, Actions=None, g_index=None, max_iter_gmr = 50):
        if example == "mean" or example == "marginal":
            # print('cluster mean')
            return clustering_mean(States, N, T, K, changepoints, Actions=None, g_index=None, max_iter_gmr=None)
        # elif example =="marginal":
        #     return clustering_marginal_dis(States, N, T, K, changepoints, Actions, g_index, max_iter_gmr)
        elif example == "cdist":
            # print('cluster dis')
            # print("N",N)
            return gmr(States, N, T, K, changepoints, Actions, g_index, max_iter_gmr)

    if example == "mean":
        cusum_forward = np.cumsum(States, axis = 1)/(np.tile((range(1,T+1)), [N, 1]).reshape([N, T, p]))
        cusum_backward = np.flip(np.cumsum(np.flip(States, axis=1),axis = 1)/(np.tile((range(1,T+1)), [N, 1]).reshape([N, T, p])), axis = 1)
    else:
        cusum_forward = None
        cusum_backward = None
        #%%
    # fit
    if init == "changepoints":
        result = changepointsNclustering(example, clustering, changepoint_detect,
                                         States, Actions, N, T, p,epsilon,kappa_max, kappa_min, K,
                                         cusum_forward,cusum_backward, C1, C2, C3,
                                         max_iter, max_iter_gmr, nthread, C,
                                         changepoints_init, clustering_warm_start,
                                         loss_path,
                                         threshold_type, B, nthread_B)

    else:
        result = clusteringNchangepoints(example, clustering, changepoint_detect,
                                         States, Actions, N, T, p, epsilon,kappa_max, kappa_min,
                                         K, cusum_forward, cusum_backward, C1, C2, C3,
                                         max_iter, init_cluster_range, nthread, C,
                                         g_index_init,clustering_warm_start,
                                         loss_path,
                                         threshold_type, B, nthread_B,init_cluster_method,distance_metric, linkage)

    return result

def fit_tuneK(K_list, States, Actions, example = "mean", init = "changepoints", kappa_max = None, kappa_min=None,epsilon=0.1,
        C1=1, C2=1/2, C3=2, alpha = 0.05, df=None, max_iter=5, init_cluster_range=None,
        max_iter_gmr = 50, seed = 1, nthread=0, C=1,changepoints_init=None,
        g_index_init_list = None,clustering_warm_start=1, loss_path =0,
        threshold_type="Chi2", B = 10, nthread_B= None,init_cluster_method = 'kmeans',distance_metric="correlation", linkage = "average"):
    '''
    Tuning the best K for clustering initialization from a list

    #param: K_list: list object
    #param: C: constant for information critirion
    '''
    IC_max = 0
    K_max = None
    res = []
    IC_model = []
    loss_model = []
    #%%
    for K in K_list:
        # print("K",K)

        out = fit(States, Actions, example, init, kappa_max,kappa_min, epsilon, K,
                C1, C2, C3, alpha, df, max_iter, init_cluster_range,
                max_iter_gmr, seed, nthread, C,changepoints_init, g_index_init_list[K_list.index(K)],
                clustering_warm_start, loss_path,threshold_type, B, nthread_B,init_cluster_method,distance_metric, linkage)

        res.append(out)
        IC_model.append(out.IC)
        loss_model.append(out.loss)
        if K == K_list[0]:
            IC_max = out.IC
            K_max = K
        if out.IC > IC_max:
            IC_max = out.IC
            K_max = K
            #%%
    tunningres = namedtuple("tunningres", ["K", "IC", "best_model", "models",
                                           "IC_model", "loss_model"])
    return tunningres(K_max, IC_max, res[K_list.index(K_max)], res, IC_model,
                      loss_model)