"""
Detect change points and clusters in mean.
"""
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
from collections import namedtuple
from scipy.linalg import block_diag
from scipy.stats import chi2, norm
from joblib import Parallel, delayed

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
        print('m',m)
        for i in range(N):
            err = []
            for k in range(K):
                err.append(np.linalg.norm((yi[i] - mean[k]), ord=2)**2/var[k])
            g_index_new[i] = err.index(min(err))
        print('g_index_new',g_index_new)
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

def gmr(States, N, T, K, changepoints,Actions=None, g_index=None, max_iter_gmr = 50):
    '''
    mixutre regression model for normal conditional distribution
    Parameters
    ----------
    g_index : the initial group index, optional
        DESCRIPTION. The default is None.
    -------
    '''
    if g_index is None:
        g_index = np.random.choice(range(K), size = N)
    g_index = g_index.astype(int)
    # mat_list = [np.array([]) for i in range(K)]
    mat_list = [[] for i in range(K)]
    y = [[] for i in range(K)]
    Xi = [[] for i in range(N)]
    yi = [[] for i in range(N)]
    for i in range(N):
        g = g_index.item(i)
        mat_tmp = np.hstack([np.ones([T -  changepoints.item(i) - 1,1]), States[i, changepoints.item(i):-1, :],
                            Actions[i, changepoints.item(i):].reshape([T -changepoints.item(i)- 1,1]), 
                            States[i,changepoints.item(i):-1, :] * (Actions[i, changepoints.item(i):].reshape([T -changepoints.item(i)- 1,1]))]) # amend this if Sit is multivairate
        
        mat_list[g].extend(mat_tmp.tolist())
        y[g].extend(States[i, changepoints.item(i)+1:,:].tolist())
        
        Xi[i] = mat_tmp
        yi[i] = States[i, changepoints.item(i)+1:,:]
    X = block_diag(*mat_list)
    y = np.vstack(y)
    reg = LinearRegression(fit_intercept=False)
    res=reg.fit(X, y)
    g_index_new = np.zeros(N, dtype=int)
    # iteration begin
    for m in range(max_iter_gmr):
        for i in range(N):
            Xistack = np.kron(np.eye(K,dtype=int),Xi[i])
            yhat = res.predict(Xistack)
            err = []
            t = Xi[i].shape[0]
            for k in range(K):
                err.append(np.linalg.norm((yi[i] - yhat[k*t + 0:k*t + t]), ord=2))
            g_index_new[i] = err.index(min(err))
        if np.prod(g_index == g_index_new):
            break
        else:
            g_index = g_index_new
            mat_list = [[] for i in range(K)]
            y = [[] for i in range(K)]
            for i in range(N):
                g = g_index.item(i)
                mat_tmp = np.hstack([np.ones([T -  changepoints.item(i) - 1,1]), States[i, changepoints.item(i):-1, :],
                                    Actions[i, changepoints.item(i):].reshape([T -changepoints.item(i)- 1,1]), 
                                    States[i,changepoints.item(i):-1, :] * (Actions[i, changepoints.item(i):].reshape([T -changepoints.item(i)- 1,1]))]) # amend this if Sit is multivairate                
                mat_list[g].extend(mat_tmp.tolist())
                y[g].extend(States[i, changepoints.item(i)+1:,:].tolist())
            X = block_diag(*mat_list)
            y = np.vstack(y)
            reg = LinearRegression(fit_intercept=False)
            res=reg.fit(X, y)
    return g_index
        
    
#%% changepoint detection function
def changemean_detect(g_index, States, N, T, kappa, epsilon,Actions=None, cusum_forward=None,
                      cusum_backward=None, C1=None, C2=None, alpha = 0.05, df=5, nthread=0):
    '''
    detect change in means
    '''
    K = len(set(g_index))
    # K = kmeans.cluster_centers_.shape[0]
    tauk = np.zeros(K)
    changepoints = np.zeros(N).reshape(-1,1)
    maxcusum_list = np.zeros(K)
    for k in range(K):
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

def changemarginal_detect(g_index, States, N, T,  kappa, epsilon, Actions=None, 
                              cusum_forward=None, cusum_backward=None, C1=None, 
                              C2=None, alpha = 0.05, df = 5, nthread=0):
    '''
    detect change in marginal distribution
    '''
    K = len(set(g_index))    
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
       var1 = (np.var(y1) * np.size(y1)+np.var(y2)*np.size(y2))/(T*np.sum(g_index == k))
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
    
def changedistribution_detect(g_index, States, N, T,  kappa, epsilon, Actions=None, 
                              cusum_forward=None, cusum_backward=None, C1=None, 
                              C2=None, alpha = 0.05, df = 5, nthread=0):
    '''
    detect change in conditional distribution
    '''
    K = len(set(g_index))    
    tauk = np.zeros(K)
    changepoints = np.zeros(N).reshape(-1,1)
    maxcusum_list = np.zeros(K)
    def run_one(u):
       X1 = []
       y1 = []
       X2 = []
       y2 = []
       X = []
       y = []
       for i in range(int(N)):
           if g_index[i] == k:
               mat_tmp = np.hstack([np.ones([u,1]), States[i, :u, :],
                                   Actions[i, :u].reshape([u,1]), 
                                   States[i, :u, :] * (Actions[i, :u].reshape([u,1]))]) # amend this if Sit is multivairate
               X1.append(mat_tmp)
               y1.append(States[i, 1:u+1,:])
               mat_tmp = np.hstack([np.ones([T -u - 1,1]), States[i, u:-1, :],
                                   Actions[i, u:].reshape([T -u - 1,1]), States[i, u:-1, :] * (Actions[i, u:].reshape([T -u - 1,1]))]) # amend this if Sit is multivairate
               X2.append(mat_tmp)
               y2.append(States[i, u+1:,:])
               mat_tmp = np.hstack([np.ones([T-1,1]), States[i, :-1, :],
                                   Actions[i, :].reshape([T-1,1]), 
                                   States[i, :-1, :] * (Actions[i, :].reshape([T-1,1]))]) # amend this if Sit is multivairate
               X.append(mat_tmp)
               y.append(States[i, 1:,:])
       X1 = np.vstack(np.array(X1)) 
       y1 = np.vstack(np.array(y1))
       X2 = np.vstack(np.array(X2)) 
       y2 = np.vstack(np.array(y2)) 
       X = np.vstack(np.array(X))
       y = np.vstack(np.array(y))
       reg0 = LinearRegression(fit_intercept=False)
       res0 = reg0.fit(X, y)
       reg1 = LinearRegression(fit_intercept=False)
       res1=reg1.fit(X1, y1)
       reg2 = LinearRegression(fit_intercept=False)
       res2=reg2.fit(X2, y2)   
       var0 = np.linalg.norm(y - res0.predict(X), ord=2)**2/N
       var1 = (np.log(np.linalg.norm(y1 - res1.predict(X1), ord=2)**2) 
            +np.log(np.linalg.norm(y2 - res2.predict(X2),ord=2)**2))/N
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
    for k in range(K):
        maxcusum = 0
        cusum_list = []
        if nthread !=0: # parallel
            res = Parallel(n_jobs=nthread)(delayed(run_one)(u) for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1))
            tauk[k] = int(T-epsilon*T)-1- res.index(np.max(res))
            maxcusum= np.max(res)
        else: # do not parallel
            for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1):
                cusum_tmp = run_one(u)
                # print('k',k,'u',u)
                # X1 = []
                # y1 = []
                # X2 = []
                # y2 = []
                # X = []
                # y = []
                # for i in range(int(N)):
                #     if g_index[i] == k:
                #         mat_tmp = np.hstack([np.ones([u,1]), States[i, :u, :],
                #                             Actions[i, :u].reshape([u,1]), 
                #                             States[i, :u, :] * (Actions[i, :u].reshape([u,1]))]) # amend this if Sit is multivairate
                #         X1.append(mat_tmp)
                #         y1.append(States[i, 1:u+1,:])
                #         mat_tmp = np.hstack([np.ones([T -u - 1,1]), States[i, u:-1, :],
                #                             Actions[i, u:].reshape([T -u - 1,1]), States[i, u:-1, :] * (Actions[i, u:].reshape([T -u - 1,1]))]) # amend this if Sit is multivairate
                #         X2.append(mat_tmp)
                #         y2.append(States[i, u+1:,:])
                #         mat_tmp = np.hstack([np.ones([T-1,1]), States[i, :-1, :],
                #                             Actions[i, :].reshape([T-1,1]), 
                #                             States[i, :-1, :] * (Actions[i, :].reshape([T-1,1]))]) # amend this if Sit is multivairate
                #         X.append(mat_tmp)
                #         y.append(States[i, 1:,:])
                # X1 = np.vstack(np.array(X1)) 
                # y1 = np.vstack(np.array(y1))
                # X2 = np.vstack(np.array(X2)) 
                # y2 = np.vstack(np.array(y2)) 
                # X = np.vstack(np.array(X))
                # y = np.vstack(np.array(y))
                # reg0 = LinearRegression(fit_intercept=False)
                # res0 = reg0.fit(X, y)
                # reg1 = LinearRegression(fit_intercept=False)
                # res1=reg1.fit(X1, y1)
                # reg2 = LinearRegression(fit_intercept=False)
                # res2=reg2.fit(X2, y2)   
                # var0 = np.linalg.norm(y - res0.predict(X), ord=2)**2/(T*np.sum(g_index == k))
                # var1 = (np.log(np.linalg.norm(y1 - res1.predict(X1), ord=2)**2) 
                #      +np.log(np.linalg.norm(y2 - res2.predict(X2),ord=2)**2))/(T*np.sum(g_index == k))
                # mean0 = res0.predict(X)
                # mean1 = res1.predict(X1)
                # mean2 = res2.predict(X2)
                # L0 = norm.pdf(y, loc = mean0, scale = np.sqrt(var0))
                # L1 = np.vstack([norm.pdf(y1, loc = mean1, scale = np.sqrt(var1)),norm.pdf(y2, loc = mean2, scale = np.sqrt(var1))])
                # L0[np.where(L0 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
                # L1[np.where(L1 < np.finfo(np.float64).tiny)] = np.finfo(np.float64).tiny
                # logL0 = np.sum(np.log(L0))
                # logL1 = np.sum(np.log(L1))
                # cusum_tmp = -2*(logL0 - logL1)
                # print(res2.coef_)
                # print(res1.coef_)
                # # cusum_tmp = -2*(np.log(np.linalg.norm(y - res0.predict(X), ord=2)**2) 
                # #     -np.log(np.linalg.norm(y1 - res1.predict(X1), ord=2)**2) 
                # #          - np.log(np.linalg.norm(y2 - res2.predict(X2),ord=2)**2))  
                # print(cusum_tmp)
                # print('** u',u, 'cusum',cusum_tmp)
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
        if maxcusum < chi2.ppf(1-alpha, df):
            tauk[k] = int(T - kappa + epsilon*T) + 1
        changepoints[(g_index == k), :] = tauk[k]   
    changepoints = changepoints.astype(int)
    return [changepoints, maxcusum_list, tauk]

#%% loops
def clusteringNchangepoints(example, clustering, changepoint_detect, States, 
                            Actions, N, T, p, epsilon, kappa, K, cusum_forward, 
                            cusum_backward, C1=1, C2=1/2, max_iter=30, 
                            init_cluster_range=None, nthread=0):
    if init_cluster_range is None:
        init_cluster_range = int(3*T/4) - 1
    changepoints_0 =  np.tile(init_cluster_range, N)
    g_index_0 = clustering(States=States,Actions=Actions,example=example, N=N, T=T, K=K, 
                           changepoints=changepoints_0)
    changepoint_list = np.zeros([N, max_iter])
    changepoint_list[:, [0]] = changepoints_0.reshape(N, 1)
    iter_num=0
    for m in range(1, max_iter):
        out=changepoint_detect(g_index = g_index_0,States=States, Actions=Actions,example=example, N=N, T=T, kappa=kappa,
                               epsilon=epsilon, cusum_forward=cusum_forward, 
                               cusum_backward=cusum_backward, C1=C1, C2=C2,nthread=nthread)
        changepoints = np.array(out[0])
        if m == 1:
            g_index = clustering(States=States, Actions=Actions,example=example, 
                                 N=N, T=T, K=K,changepoints=changepoints_0)
        else:
            g_index = clustering(States=States, Actions=Actions,example=example, g_index=g_index,
                                 N=N, T=T, K=K,changepoints=changepoints_0)
        changepoint_list[:, [m]] = changepoints.reshape(N, 1)
        if np.prod(changepoints == changepoints_0):
            iter_num = m
            break
        else:
            changepoints_0 = changepoints
            g_index_0 = g_index
        iter_num = m
    out=changepoint_detect(g_index = g_index_0,States=States, Actions=Actions,example=example, N=N, T=T, kappa=kappa,
                           epsilon=epsilon, cusum_forward=cusum_forward, 
                           cusum_backward=cusum_backward, C1=C1, C2=C2,nthread=nthread)
    changepoints = np.array(out[0])
    changepoint_list = changepoint_list[:, :(m+2)]
    result = namedtuple("result", ["iter_num", "kmeans", "changepoints", "changepoint_eachiter"])
    return result(iter_num, g_index, changepoints, changepoint_list)

def changepointsNclustering(example, clustering, changepoint_detect, States,Actions,
                            N, T, p, epsilon, kappa, K, cusum_forward, 
                            cusum_backward, C1=1, C2=1/2, 
                            max_iter=30, max_iter_gmr = 50, nthread=0):
    kmeans_0 = KMeans(n_clusters=N, random_state=0).fit(States[:,:,0])
    g_index = kmeans_0.labels_
    out = changepoint_detect(g_index = g_index,States=States, Actions=Actions, example=example, N=N, T=T, kappa=kappa, epsilon=epsilon,
                             cusum_forward=cusum_forward, cusum_backward=cusum_backward,
                             C1=C1, C2=C2, nthread=nthread)
    changepoints_0 = out[0]
    # print(changepoints_0)
    changepoint_list = np.zeros([N, max_iter])
    changepoint_list[:, [0]] = changepoints_0.reshape(N, 1)
    iter_num = 0
    for m in range(1, max_iter):
        if m == 1:
            g_index = clustering(States=States, Actions=Actions,example=example, 
                                 N=N, T=T, K=K,changepoints=changepoints_0)
        else:
            g_index = clustering(States=States, Actions=Actions,example=example, g_index=g_index,
                                 N=N, T=T, K=K,changepoints=changepoints_0)
        out=changepoint_detect(g_index = g_index,States=States, Actions=Actions, example=example,N=N, T=T, kappa=kappa, epsilon=epsilon,
                                 cusum_forward=cusum_forward, cusum_backward=cusum_backward,
                                 C1=C1, C2=C2,nthread=nthread)
        changepoints = np.array(out[0])
        # print(changepoints)
        changepoint_list[:, [m]] = changepoints.reshape(N, 1)
        if m != 1 and np.prod(changepoints == changepoints_0):
            iter_num = m
            break
        else:
            changepoints_0 = changepoints
        iter_num = m
    changepoint_list = changepoint_list[:, :(m+1)]
    # print(changepoints_0)
    result = namedtuple("result", ["iter_num", "kmeans", "changepoints", "changepoint_eachiter"])
    return result(iter_num, g_index, changepoints_0, changepoint_list)
       
#%% fit
def fit(States, Actions, example = "mean", init = "changepoints", kappa = None, epsilon=0.1, K=2, 
        C1=1, C2=1/2, alpha = 0.05, df=4, max_iter=30, init_cluster_range=None, 
        max_iter_gmr = 50, seed = 1, nthread=0):
    '''
    :param example: "mean", "conditional dis"
    :param inti: initial estimator, "changepoints", detect changepoints for each trajectrory separately, "clusters", kemans
    '''
    np.random.seed(seed)
    N = States.shape[0]
    T = States.shape[1]
    p = States.shape[2]
    if kappa is None:
        kappa = T
        
    def changepoint_detect(g_index, States, N, T,  kappa, epsilon, example, Actions=Actions, 
                              cusum_forward=None, cusum_backward=None, C1=C1, 
                              C2=C2, alpha = alpha, df = df,nthread=nthread):
        if example == "mean":
            # print('cluster mean')
            return changemean_detect(g_index, States, N, T,  kappa, epsilon, Actions=None, 
                                 cusum_forward=cusum_forward, 
                                 cusum_backward=cusum_backward, C1=C1 , C2=C2, 
                                 alpha=None, df=None,nthread=nthread)
        elif example =="marginal":
            # print("marginal")
            return changemarginal_detect(g_index, States, N, T,  kappa, epsilon, Actions=Actions, 
                                         cusum_forward=None, cusum_backward=None,
                                         C1=C1, C2=C2, alpha=alpha, df=df,nthread=nthread)
        elif example == "cdis":
            return changedistribution_detect(g_index, States, N, T,  kappa, epsilon, Actions=Actions, 
                                         cusum_forward=None, cusum_backward=None,
                                         C1=None, C2=None, alpha=alpha, df=df,nthread=nthread)
    def clustering(States, N, T, K, changepoints,example, Actions=None, g_index=None, max_iter_gmr = 50):
        if example == "mean" or example == "marginal":
            # print('cluster mean')
            return clustering_mean(States, N, T, K, changepoints, Actions=None, g_index=None, max_iter_gmr=None)
        # elif example =="marginal":
        #     return clustering_marginal_dis(States, N, T, K, changepoints, Actions, g_index, max_iter_gmr)
        elif example == "cdis":
            # print('cluster dis')
            return gmr(States, N, T, K, changepoints, Actions, g_index, max_iter_gmr)
    
    if example == "mean":
        cusum_forward = np.cumsum(States, axis = 1)/(np.tile((range(1,T+1)), [N, 1]).reshape([N, T, p]))
        cusum_backward = np.flip(np.cumsum(np.flip(States, axis=1),axis = 1)/(np.tile((range(1,T+1)), [N, 1]).reshape([N, T, p])), axis = 1)
    else:
        cusum_forward = None
        cusum_backward = None
    # fit                
    if init == "changepoints":
        result = changepointsNclustering(example, clustering, changepoint_detect, States, Actions, N, T, p,epsilon, kappa, K, cusum_forward,
                                                                                   cusum_backward, C1, C2, max_iter, max_iter_gmr, nthread)
        # result[2]
    else:
        result = clusteringNchangepoints(example, clustering, changepoint_detect, States, Actions, N, T, p, epsilon, kappa, K, cusum_forward,
                                                                                    cusum_backward, C1, C2, max_iter, init_cluster_range, nthread) 
        
    return result
    
