"""
Detect change points and clusters in mean.
"""
from sklearn.cluster import KMeans
import numpy as np

def clustering(States, N, T, p, K, changepoints):
    X = np.zeros([N, p])
    for i in range(N):
        X[i, 0] = np.mean(States[i, changepoints[i].astype(int).item():T, :])
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    return kmeans

def changepoint_detect(kmeans, N, T, kappa, epsilon, cusum_forward, cusum_backward, C1, C2):
    K = kmeans.cluster_centers_.shape[0]
    tauk = np.zeros(K)
    changepoints = np.zeros(N).reshape(-1,1)
    maxcusum_list = np.zeros(K)
    for k in range(K):
        maxcusum = 0
        cusum_list = np.zeros(int(T-epsilon*T)-1- int(T - kappa + epsilon*T)) 
        z=0
        for u in range(int(T-epsilon*T)-1, int(T - kappa + epsilon*T), -1):
            print('u',u, 'z',z)
            cusum_tmp = (np.sqrt((T-u)*(u-T+kappa)/(kappa**2))*np.sum(kmeans.labels_ == k) 
                         * np.linalg.norm(
                             np.dot((kmeans.labels_ == k).reshape(1, N), 
                                    (cusum_forward[:, u] - cusum_backward[:, u+1]).reshape(N,1)), ord = 2))
            cusum_list[z] = cusum_tmp
            z = z+1
            if u == int(T-epsilon*T)-1:
                maxcusum = cusum_tmp
                maxcusum_list[k] = maxcusum
                tauk[k] = u
            elif maxcusum <= cusum_tmp:
                print('** maxcusum',maxcusum)
                maxcusum = cusum_tmp
                maxcusum_list[k] = maxcusum
                tauk[k] = u
        if maxcusum < C1*(kappa * np.sum(kmeans.labels_ == k))**(-C2) * np.sqrt(np.log(N * T)):
            tauk[k] = int(T - kappa + epsilon*T) + 1
        changepoints[(kmeans.labels_ == k), :] = tauk[k]
        # print('k', k, 'tauk', tauk[k])
    return [changepoints, maxcusum_list, tauk]
            
def clusteringNchangepoints(States, N, T, p, epsilon, kappa, K, cusum_forward, cusum_backward, C1=1, C2=1/2, max_iter=30, init_cluster_range=None):
    if init_cluster_range is None:
        init_cluster_range = int(3*T/4) - 1
    changepoints_0 =  np.tile(init_cluster_range, N)
    kmeans_0 = clustering(States, N, T, p, K, changepoints_0)
    changepoint_list = np.zeros([N, max_iter])
    changepoint_list[:, [0]] = changepoints_0.reshape(N, 1)
    iter_num=0
    for m in range(1, max_iter):
        out=changepoint_detect(kmeans_0, N, T, kappa, epsilon, cusum_forward, cusum_backward, C1, C2)
        changepoints = np.array(out[0])
        kmeans = clustering(States, N, T, p, K, changepoints) 
        changepoint_list[:, [m]] = changepoints.reshape(N, 1)
        if np.prod(changepoints == changepoints_0):
            iter_num = m
            break
        else:
            changepoints_0 = changepoints
            kmeans_0 = kmeans
        iter_num = m
    return [iter_num, kmeans, changepoints, changepoint_list]

def changepointsNclustering(States, N, T, p, epsilon, kappa, K, cusum_forward, cusum_backward, C1=1, C2=1/2, max_iter=30):
    kmeans_0 = KMeans(n_clusters=N, random_state=0).fit(States[:,:,0])
    out = changepoint_detect(kmeans_0, N, T, kappa, epsilon, cusum_forward, cusum_backward, C1, C2)
    changepoints_0 = out[0]
    changepoint_list = np.zeros([N, max_iter])
    changepoint_list[:, [0]] = changepoints_0.reshape(N, 1)
    iter_num = 0
    for m in range(1, max_iter):
        kmeans = clustering(States, N, T, p, K,changepoints_0)
        out=changepoint_detect(kmeans, N, T, kappa, epsilon, cusum_forward, cusum_backward, C1, C2)
        changepoints = np.array(out[0])
        changepoint_list[:, [m]] = changepoints.reshape(N, 1)
        if m != 1 and np.prod(changepoints == changepoints_0):
            iter_num = m
            break
        else:
            changepoints_0 = changepoints
            kmeans_0 = kmeans
        iter_num = m
    return [iter_num, kmeans, changepoints, changepoint_list]
       
def detect_mean(States, init = "changepoints", kappa = None, epsilon=0.1, K=2, C1=1, C2=1/2, max_iter=30, init_cluster_range=None):
    '''
    :param inti: initial estimator, "changepoints", detect changepoints for each trajectrory separately, "clusters", kemans
    '''
    N = States.shape[0]
    T = States.shape[1]
    p = States.shape[2]
    if kappa is None:
        kappa = T
    cusum_forward = np.cumsum(States, axis = 1)/(np.tile((range(1,T+1)), [N, 1]).reshape([N, T, p]))
    cusum_backward = np.flip(np.cumsum(np.flip(States, axis=1),axis = 1)/(np.tile((range(1,T+1)), [N, 1]).reshape([N, T, p])), axis = 1)
    States2 = np.square(States)
    
    if init == "changepoints":
        iter_num, kmeans, changepoints, changepoint_list = changepointsNclustering(States, N, T, p, epsilon, kappa, K, cusum_forward,
                                                                                   cusum_backward, C1, C2, max_iter)
    else:
        iter_num, kmeans, changepoints, changepoint_list = clusteringNchangepoints(States, N, T, p, epsilon, kappa, K, cusum_forward,
                                                                                   cusum_backward, C1, C2, max_iter, init_cluster_range) 
    return [iter_num, kmeans, changepoints, changepoint_list]
    
