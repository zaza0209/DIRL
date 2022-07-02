import platform, sys, os, pickle, re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import simu.compute_test_statistics as stat
import simu.simulate_data_pd as sim
import simu.simu_mean_detect as mean_detect
import simu.utilities as uti
from sklearn.metrics.cluster import adjusted_rand_score
import pprint
import pandas as pd
#%% IC
def h_in_IC(changepoints,T):
    '''
    The bonus term for changepoints location in IC
    '''
    return (np.sum(T-1 -changepoints)/np.log(np.sum(T-1 - changepoints)))

def IC(loss, changepoints, g_index, N, T, K, C):
    # """
    # Information criterion
    # Parameters
    # ----------
    # loss : TYPE
    #     DESCRIPTION.
    # C : h(\sum \tau_i) = C \sum \tau_i/log(\sum \tau_i)

    # """
    Kl = K*np.log(np.sum(T-1 -changepoints))
    # print("kl", Kl)
    Ck, indicesList, occurCount = np.unique(g_index, return_index = True,return_counts=True)
    # print("C",C)
    # print('c',occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * C * np.log(np.sum(T-1 -changepoints)))
    # # # print('loss', loss)
    # print('occurCount',occurCount)
    # print('(T-1 -changepoints)[np.s_[indicesList]]',(T-1 -changepoints)[np.s_[indicesList]])
    # print('np.log(np.sum(T-1 -changepoints)',np.log(np.sum(T-1 -changepoints)))
    # print("ic",loss - Kl+ occurCount.dot((T-1 -changepoints)[np.s_[indicesList]]) * C * np.log(np.sum(T-1 -changepoints)))
    return loss - Kl+ occurCount.dot((T-1 -changepoints)[np.s_[indicesList]])/(N*T) * C * h_in_IC(changepoints, T)

def paramInIC(model, N, K, T, include_path_loss =0):
    if not include_path_loss:
        Kl_mat =  K*np.log(np.sum(T-1 -model['changepoint']))
        Ck, indicesList, occurCount = np.unique(model['group'], return_index = True,return_counts=True)
        c_mat = occurCount.dot((T-1 -model['changepoint'])[np.s_[indicesList]])/(N*T) * h_in_IC(model['changepoint'], T)
        print("occurCount.dot((T-1 -model['changepoint'])[np.s_[indicesList]])/(N*T)",occurCount.dot((T-1 -model['changepoint'])[np.s_[indicesList]])/(N*T),"h_in_IC(model['changepoint'], T)",h_in_IC(model['changepoint'], T))
        print(c_mat)
        return Kl_mat, c_mat
    else:
        Kl_mat_list = []
        c_mat_list =[]
        cp_eachiter_rev = model['changepoint_eachiter'][:, range(model['changepoint_eachiter'].shape[1]-1,-1,-1)]
        g_eachiter_rev = model['g_index_eachiter'][:, range(model['g_index_eachiter'].shape[1]-1,-1,-1)]
        for ii in range(min(model['changepoint_eachiter'].shape[1], model['g_index_eachiter'].shape[1])):
            Kl_mat_list.append(K*np.log(np.sum(T-1 - cp_eachiter_rev[:,ii])))
            Ck, indicesList, occurCount = np.unique(g_eachiter_rev[:,ii], return_index = True,return_counts=True)
            c_mat_list.append(occurCount.dot((T-1 -cp_eachiter_rev[:,ii])[np.s_[indicesList]])/(N*T) * h_in_IC(cp_eachiter_rev[:,ii], T))
        return Kl_mat_list[::-1], c_mat_list[::-1]
# def multi_init:
#     init_list = ["changepoints_ora", "clustering_ora","changepoint_separ", "changepoint_no", "changepoints_random", 'randomclustering']
#     # init_list = ["changepoints_ora", "clustering_ora",'randomclustering']
#     C_list = np.arange(0,10,1).tolist()
#     K_list = range(2, 6)
#     ic_list = [None] * len(init_list)
#     loss_mat = [None] * len(init_list)
#     Kl_mat = [None] * len(init_list)
#     c_mat = [None] * len(init_list)
#     for init in init_list:
#         if init == "randomclustering":
#             loss_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
#             Kl_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
#             c_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
#             for seed in range(M):
#                 for K in K_list:
#                     setpath(trans_setting, K = K, init=init)
#                     file_name = "seed_"+str(seed)+".dat"
#                     pkl_file = open(file_name, 'rb')
#                     t = pickle.load(pkl_file)
#                     # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
#                     #                               t['group'], N, T, K, C)
#                     loss_mat[init_list.index(init)][seed, K_list.index(K)] = t['loss']# / np.mean(T - t['changepoint'] - 1)
#                     # Kl_mat[init_list.index(init)][seed, K_list.index(K)] =  K*np.log(np.sum(T-1 -t['changepoint']))
#                     # Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
#                     # c_mat[init_list.index(init)][seed, K_list.index(K)] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
#                     pkl_file.close()
#                     Kl_mat[init_list.index(init)][seed, K_list.index(K)], c_mat[init_list.index(init)][seed, K_list.index(K)]=uti.paramInIC(t, N, K, T)
#         else:
#             loss_mat[init_list.index(init)] = np.zeros([M, 1])
#             Kl_mat[init_list.index(init)] = np.zeros([M, 1])
#             c_mat[init_list.index(init)] = np.zeros([M, 1])
#             for seed in range(M):
#                 setpath(trans_setting, K = 3, init=init)
#                 file_name = "seed_"+str(seed)+".dat"
#                 pkl_file = open(file_name, 'rb')
#                 t = pickle.load(pkl_file)
#                 # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
#                 #                               t['group'], N, T, K, C)
#                 # loss = mean_detect.goodnessofClustering(States, N, T, 3, t['changepoint'], Actions, t['group'])
#                 loss_mat[init_list.index(init)][seed, 0] = t['loss'] #/ np.mean(T - t['changepoint'] - 1)
#                 # Kl_mat[init_list.index(init)][seed, 0] =  K*np.log(np.sum(T-1 -t['changepoint']))
#                 # Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
#                 # c_mat[init_list.index(init)][seed, 0] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
#                 pkl_file.close()
#                 Kl_mat[init_list.index(init)][seed, 0],c_mat[init_list.index(init)][seed, 0] = uti.paramInIC(t, N, 3, T)



#     res_diffC_all = [None] * len(C_list)
#     for C in C_list:
#         init_count = np.zeros(len(init_list))
#         K_count = np.zeros(len(K_list))
#         bestK_list = np.zeros(M)
#         changepoint_err_list = []
#         cluster_err_list= []
#         iter_num =[]
#         for seed in range(M):
#             # print('seed', seed)
#             for init in init_list:
#                 # print('====init:', init,"====")
#                 if init == "randomclustering":
#                     # print('loss_mat[',init_list.index(init),'][seed, :]',loss_mat[init_list.index(init)][seed, :], 'Kl', Kl_mat[init_list.index(init)][seed, :] ,'cmat', c_mat[init_list.index(init)][seed,:]*C)
#                     # for C in np.arange(10.433, 10.44, 0.00001):
#                         # print('====C',C,"=====")
#                         # print(loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C)
#                         # print('**maxK', K_list[np.argmax(loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C)])
#                     # loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C
#                     ic_list[init_list.index(init)] = loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C
#                     clusterK = K_list[np.argmax(ic_list[init_list.index('randomclustering')])]
#                     # print('ic_list[init_list.index("randomclustering")]',ic_list[init_list.index('randomclustering')])
#                     ic_list[init_list.index('randomclustering')] = np.max(ic_list[init_list.index('randomclustering')])
#                 else:    
#                     # print('loss_mat[',init_list.index(init),'][seed, :]',loss_mat[init_list.index(init)][seed, :], 'Kl', Kl_mat[init_list.index(init)][seed, :] ,'cmat', c_mat[init_list.index(init)][seed,:]*C)
#                     ic_list[init_list.index(init)] = (loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C)[0]
#             # print('ic_list' ,ic_list)
#             init = init_list[np.where(ic_list == np.max(ic_list))[0][0]]
#             init_count[np.where(ic_list == np.max(ic_list))[0][0]] = init_count[np.where(ic_list == np.max(ic_list))[0][0]] + 1
#             if init == "randomclustering":
#                 bestK = clusterK
#             else:
#                 bestK = 3
#             K_count[K_list.index(bestK)] = K_count[K_list.index(bestK)] +1
#             # print('**** init', init, 'K', bestK)
#             # bestK = K_list[np.where(ic_list == np.max(ic_list))[1][0]]
#             bestK_list[seed]=bestK
#             setpath(trans_setting, K = bestK, init=init)
#             file_name = "seed_"+str(seed)+".dat"
#             pkl_file = open(file_name, 'rb')
#             t = pickle.load(pkl_file)
#             print(t['changepoint'])
#             changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
#                                                     t['changepoint'].squeeze(), N, T)
#             changepoint_err_list.append(changepoint_err)
#             # print('cp_err_best', changepoint_err, "ARI", cluster_err)
#             cluster_err_list.append(cluster_err)
#             iter_num.append(t['iter_num'])
#             pkl_file.close()
            
#         res_diffC_all[C_list.index(C)] = {"C":C,
#                                       "changepoint_err": np.mean(changepoint_err_list),
#                                       "ARI": np.mean(cluster_err_list),
#                                       "iter_num":np.mean(iter_num),
#                                       "bestK":np.mean(bestK_list),
#                                       "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
#                                       'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
#                                       'iter_num_var':np.std(iter_num)/np.sqrt(M),
#                                       'bestK_var':np.std(bestK_list)/np.sqrt(M),
#                                       "init_count":init_count,
#                                       "K_count":K_count}


#%% time series clustering
from dtaidistance import dtw
from scipy.cluster.hierarchy import single, complete, average, ward, fcluster
# from scipy.cluster.hierarchy import fcluster
# import pandas as pd
def my_hierachy(States, K, distance_metric='correlation', linkage = "average"):

    if distance_metric == "correlation":
        # Reshape the data so each series is a column and call the dataframe.corr() function 
        # distance_matrix = pd.concat([series for series in X['dim_0'].values], axis=1).corr()
        distance_matrix = pd.DataFrame(np.vstack(States[:,:,:].reshape(States.shape[0], 1, -1)).T).corr()
    elif distance_metric == "DWT":
        # Italy Power Demand time series are loaded in a pd.Series format.
        # The dtw_distance function expects series to be shaped as a (l, m) array, 
        # where l=length of series, m=# dimensions           
        # series_list = X['dim_0'].values
        series_list = [None] * States.shape[0]
        for i in range(len(series_list)):
            # length = len(series_list[i])
            series_list[i] = States[i,:,:].flatten().reshape(-1, 1)
        
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
    #>> 10 unique clusters
    
    # hand-select an appropriate cut-off on the dendrogram
    # cluster_labels = fcluster(linkage_matrix, 600, criterion='distance')
    # print(np.unique(cluster_labels))
    # #>> 3 unique clusters
    # cluster_labels = fcluster(linkage_matrix, 800, criterion='distance')
    # print(np.unique(cluster_labels))
    #>> 2 unique clusters
    cluster_labels = (cluster_labels-1).astype(int)
    return cluster_labels
