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
#%%
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
#%% 
# def gen_dat(N, T, K, coef, signal, changepoint_list=changepoint_list,
#             trans_setting="pwsonst2", seed=1):
#     np.random.seed(seed)
#     if changepoint_list is None:
#         changepoint_list = [int(T/2) +30 + int(0.1 * T) - 1, int(T/2)-1 +30, int(T/2) - int(0.1 * T) +30- 1] 
#         # changepoint_list = [9, 49, 89] 
#         # changepoint_list = [19, 49, 79] 

#     changepoints_true = np.zeros([N, 1])
#     States = np.zeros([N, T, p])
#     Rewards = np.zeros([N, T-1])
#     Actions = np.zeros([N, T-1])
#     def myreward_function(t):
#         return sim_dat.reward_homo()
#     coef_tmp = [None] * 2
#     changepoint = 0
#     for i in range(N):
#         if i < int(N/4):
#             changepoint = changepoint_list[0]
#             coef_tmp[0] = coef[0][0]
#             coef_tmp[1] = coef[1][0]
#             signal_tmp = [signal[0][0], signal[1][0]]
#             # print('signal_tmp',signal_tmp)
#         elif i < int(N/3):
#             changepoint = changepoint_list[0]
#             coef_tmp[0] = coef[0][1]
#             coef_tmp[1] = coef[1][0]
#             signal_tmp = [signal[0][0], signal[1][0]]
#         elif i < int(N/2):
#             changepoint = changepoint_list[1]
#             coef_tmp[0] = coef[0][1]
#             coef_tmp[1] = coef[1][1]
#             signal_tmp = [signal[0][0], signal[1][1]]
#         elif i < int(2*N/3):
#             changepoint = changepoint_list[1]
#             coef_tmp[0] = coef[0][2]
#             coef_tmp[1] = coef[1][1]
#             signal_tmp = [signal[0][1], signal[1][1]]
#         elif i < int(3*N/4):
#             changepoint = changepoint_list[2]
#             coef_tmp[0] = coef[0][2]
#             coef_tmp[1] = coef[1][2]
#             signal_tmp = [signal[0][1], signal[1][2]]
#         else:
#             changepoint = changepoint_list[2]
#             coef_tmp[0] = coef[0][3]
#             coef_tmp[1] = coef[1][2]
#             signal_tmp = [signal[0][1], signal[1][2]]
            
#         sim_dat = sim.simulate_data(1, T, p, changepoint, delta)
#         def mytransition_function(t):
#             return sim_dat.transition_pwconstant2(t, mean, cov, coef_tmp,signal_tmp)
#         States0, Rewards0, Actions0 = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function)
#         States[i, :, :] = States0
#         Rewards[i, :] = Rewards0
#         Actions[i, :] = Actions0
#         changepoints_true[i, ] = changepoint
#     # normalize state variables
#     def transform(x):
#         return (x - np.mean(x)) / np.std(x)
#     for i in range(p):
#         States[:,:,i] = transform(States[:,:,i])
#     g_index_true = np.append([np.zeros(int(N/3)), np.ones(int(N/3))], 2*np.ones(int(N/3)))
#     return States, Rewards, Actions, changepoints_true, g_index_true

# def setpath(trans_setting, K=None, init=None):
#     os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
#     if not os.path.exists('results'):
#         os.makedirs('results', exist_ok=True)
#     # if K is not None:
#     #     path_name = 'results/2x3/K'+str(K)+'/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) +\
#     #             '_kappa' + str(kappa) + '_N' + str(N) + '_T'+ str(T)+'clusterws'+str(clustering_warm_start) + 'signal' + str(signal)
#     # else:
#     path_name = 'results/2x3/'+init+'/K'+str(K)+'/coef0'+   re.sub("\\,", "", re.sub("\\ ", "", re.sub("\\.", "", re.sub("\\]","", re.sub("\\[", "", str(coef[0]))))))+\
#     '/coef1'+   re.sub("\\ ", "",re.sub("\\.", "", re.sub("\\]","", re.sub("\\[", "", re.sub("\\, ", "", str(coef[1]))))))+\
#         '/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) +\
#                 '_N' + str(N) + '_T'+ str(T)+'clusterws'+str(clustering_warm_start) + 'signal' + str(signal) + 'cov' + str(cov)
#     if not os.path.exists(path_name):
#         os.makedirs(path_name, exist_ok=True)
#     # path_name += '/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) + \
#     #              '_kappa' + str(kappa) + '_N' + str(N) + '_1d_' + str(seed)
#     # if not os.path.exists(path_name):
#     #     os.makedirs(path_name, exist_ok=True)
#     os.chdir(path_name)

# def save_data(out, seed, K_list=None, init=None):
#     if type(K_list) is range or type(K_list) is list:
#         for k in K_list:
#             setpath(trans_setting,k,init)
#             tmp = out.models[K_list.index(k)]
#             file_name = "seed_"+str(seed)+".dat"
#             with open(file_name, "wb") as f:
#                 pickle.dump({'iter_num' : tmp.iter_num, 'group':tmp.g_index,
#                              'changepoint' :tmp.changepoints, 
#                              'changepoint_eachiter':tmp.changepoint_eachiter,
#                              'g_index_eachiter':tmp.g_index_eachiter,
#                              'loss':tmp.loss,
#                              "ic":tmp.IC,
#                              "K":k}, f)
#     else:
#         setpath(trans_setting,K_list, init)
#         tmp = out
#         file_name = "seed_"+str(seed)+".dat"
#         with open(file_name, "wb") as f:
#             pickle.dump({'iter_num' : tmp.iter_num, 'group':tmp.g_index,
#                          'changepoint' :tmp.changepoints, 
#                          'changepoint_eachiter':tmp.changepoint_eachiter,
#                          'g_index_eachiter':tmp.g_index_eachiter,
#                          'loss':tmp.loss,
#                          "ic":tmp.IC,
#                          "K":K_list}, f)
# #%% ic
# def IC_include_path_loss(init_list,C_list, N,T, M):
#     include_path_loss = 1
#     # init_list = ["changepoints_ora", "clustering_ora","changepoint_separ", "changepoint_no", "changepoints_random", 'randomclustering']
#     # init_list = ["changepoints_ora", 'randomclustering']
#     C_list = np.arange(0,10,1).tolist()
#     K_list = range(2, 6)
#     # ic_list = [None] * len(init_list)
#     loss_mat = [None] * len(init_list)
#     ic_mat = [None] * len(init_list)
#     Kl_mat = [None] * len(init_list)
#     c_mat = [None] * len(init_list)
#     for C in C_list:
#         init_count = np.zeros(len(init_list))
#         K_count = np.zeros(len(K_list))
#         bestK_list = []
#         changepoint_err_list = []
#         cluster_err_list= []
#         iter_num =[]
#         for seed in range(M):
#             best_init = None
#             bestK = None
#             bestIC = None
#             bestiter = None
#             States, Rewards, Actions, changepoints_true,g_index_true = gen_dat(N=N, T=T, K=3, coef=coef, 
#                                                                   signal =signal,changepoint_list=None,
#                                                                   trans_setting=trans_setting,seed=seed + 100)

#             for init in init_list:
#                 if init == "randomclustering":
#                     loss_mat[init_list.index(init)] = np.zeros([M, 1])
#                     ic_mat[init_list.index(init)] = np.zeros([M, 1])
#                     Kl_mat[init_list.index(init)] = np.zeros([M, 1])
#                     c_mat[init_list.index(init)] = np.zeros([M, 1])
#                     ic_randomK_eachseed = []
#                     iter_num = []
#                     for K in K_list:
#                         setpath(trans_setting, K = K, init=init)
#                         file_name = "seed_"+str(seed)+".dat"
#                         pkl_file = open(file_name, 'rb')
#                         t = pickle.load(pkl_file)
#                         pkl_file.close()
#                         loss_tmp = []
#                         Kl_tmp = []
#                         c_mat_tmp = []
#                         cp_rev = t['changepoint_eachiter'][:, range(t['changepoint_eachiter'].shape[1]-1,-1,-1)]
#                         g_rev = t['g_index_eachiter'][:, range(t['g_index_eachiter'].shape[1]-1,-1,-1)]
#                         for ii in range(min(t['changepoint_eachiter'].shape[1], t['g_index_eachiter'].shape[1])):
#                             loss_tmp.append(mean_detect.goodnessofClustering(States, N, T, K, cp_rev[:, ii], Actions, g_rev[:,ii]))
#                         loss_tmp = loss_tmp [::-1]
#                         Kl_tmp, c_mat_tmp = uti.paramInIC(t, N, K, T, include_path_loss)
#                         ic_tmp =  np.array(loss_tmp) - np.array(Kl_tmp) +  np.array(c_mat_tmp) * C
#                         iter_num.append(np.argmax(ic_tmp))
#                         ic_randomK_eachseed.append(np.max(ic_tmp))
#                     if bestIC is None:
#                         best_init = init
#                         bestIC = np.max(ic_randomK_eachseed)
#                         bestK = K_list[np.argmax(ic_randomK_eachseed)]
#                         bestiter = iter_num[np.argmax(ic_randomK_eachseed)]
#                     elif bestIC < np.max(ic_randomK_eachseed):
#                         best_init = init
#                         bestIC = np.max(ic_randomK_eachseed)
#                         bestK = K_list[np.argmax(ic_randomK_eachseed)]
#                         bestiter = iter_num[np.argmax(ic_randomK_eachseed)]
#                 else:
#                     loss_mat[init_list.index(init)] = np.zeros([M, 1])
#                     Kl_mat[init_list.index(init)] = np.zeros([M, 1])
#                     c_mat[init_list.index(init)] = np.zeros([M, 1])
#                     setpath(trans_setting, K = 3, init=init)
#                     file_name = "seed_"+str(seed)+".dat"
#                     pkl_file = open(file_name, 'rb')
#                     t = pickle.load(pkl_file)
#                     pkl_file.close()
#                     loss_tmp = []
#                     Kl_tmp = []
#                     c_mat_tmp = []
#                     cp_rev = t['changepoint_eachiter'][:, range(t['changepoint_eachiter'].shape[1]-1,-1,-1)]
#                     g_rev = t['g_index_eachiter'][:, range(t['g_index_eachiter'].shape[1]-1,-1,-1)]
#                     for ii in range(min(t['changepoint_eachiter'].shape[1], t['g_index_eachiter'].shape[1])):
#                         if len(set(g_rev[:,ii])) == 1:
#                             loss_tmp.append(-sys.float_info.max)
#                         else:
#                             loss_tmp.append(mean_detect.goodnessofClustering(States, N, T, K, cp_rev[:, ii], Actions, g_rev[:,ii]))
#                     loss_tmp = loss_tmp [::-1]
#                     Kl_tmp, c_mat_tmp = uti.paramInIC(t, N, 3, T, include_path_loss =0)
#                     ic_tmp =  np.array(loss_tmp) - np.array(Kl_tmp) +  np.array(c_mat_tmp) * C
#                     # ic_mat[init_list.index(init)][seed, 0] = np.max(ic_tmp)
#                     if bestIC is None:
#                         best_init = init
#                         bestIC = np.max(ic_tmp)
#                         bestK = 3
#                         bestiter = np.argmax(ic_tmp)
#                     elif bestIC < np.max(ic_tmp):
#                         best_init = init
#                         bestIC = np.max(ic_tmp)
#                         bestK = 3
#                         bestiter = np.argmax(ic_tmp)
                  
#             # collect the best performance for this seed
#             setpath(trans_setting, K = bestK, init=best_init)
#             file_name = "seed_"+str(seed)+".dat"
#             pkl_file = open(file_name, 'rb')
#             t = pickle.load(pkl_file)
#             bestK_list.append(bestK)
#             # print(t['changepoint'])
#             if t['changepoint_eachiter'].shape[1] < t['g_index_eachiter'].shape[1]:
#                 cp = t['changepoint_eachiter'][:, bestiter]
#                 g_index = t['g_index_eachiter'][:, bestiter+1]
#             else:
#                 cp = t['changepoint_eachiter'][:, bestiter+1]
#                 g_index = t['g_index_eachiter'][:, bestiter]
#             changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), g_index.squeeze(),
#                                                     cp.squeeze(), N, T)
#             changepoint_err_list.append(changepoint_err)
#             # print('cp_err_best', changepoint_err, "ARI", cluster_err)
#             cluster_err_list.append(cluster_err)
#             iter_num.append(t['iter_num'])
#             pkl_file.close()
#             init_count[init_list.index(best_init)] = init_count[init_list.index(best_init)] + 1
#             K_count[K_list.index(bestK)] = K_count[K_list.index(bestK)] +1
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
#     dat_all = pd.DataFrame(res_diffC_all)
#     dat_all = round(dat_all, 3)
#     for i in range(1,5):
#         dat_all.iloc[:,i] = [str(a) +'('+ str(b)+")" 
#                              for a,b in zip(dat_all.iloc[:,i],dat_all.iloc[:,i+4])]
#     col = dat_all.columns.tolist()
#     dat_all = dat_all[[col[0], col[4]]+ col[1:4]]
#     # print(dat_all)
#     return dat_all, res_diffC_all