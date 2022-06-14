#%% analyse results
# import simu.01_sim_2d_run
import numpy as np
import simu.utilities as uti
include_path_loss = 1
import simu.simu_mean_detect as mean_detect
#%% analyse results 1 -- init changepoint separately K = 3/ changepoint_no/ changepoints random
# C_list =[0, 0.1, 1, 10,50,100,200,500, 1000]
init_list = ["changepoint_separ", "changepoint_no","changepoints_random"]
K_list = [3]
ic_list = np.zeros(len(init_list))
loss_mat_cp = np.zeros([M, len(init_list)])
Kl_mat = np.zeros([M, len(init_list)])
c_mat = np.zeros([M, len(init_list)])
changepoint_err_mat_cp = np.zeros([M,len(init_list)])
cluster_err_mat_cp = np.zeros([M,len(init_list)])
iter_mat_cp = np.zeros([M,len(init_list)])

for init in init_list:
    for seed in range(M):
        for K in K_list:
            setpath(trans_setting, K = K, init=init)
            file_name = "seed_"+str(seed)+".dat"
            pkl_file = open(file_name, 'rb')
            t = pickle.load(pkl_file)
            # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
            #                               t['group'], N, T, K, C)
            loss_mat_cp[seed, init_list.index(init)] = t['loss']
            Kl_mat[seed, init_list.index(init)] =  K*np.log(np.sum(T-1 -t['changepoint']))
            Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
            c_mat[seed, init_list.index(init)] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
            changepoint_err_mat_cp[seed,init_list.index(init)],cluster_err_mat_cp[seed,init_list.index(init)] = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                    t['changepoint'].squeeze(), N, T)
            iter_mat_cp[seed,init_list.index(init)] = t['iter_num']
            pkl_file.close()


res_diffC_cp = [None] * len(C_list)
for C in C_list:
    # bestK_list = np.zeros(M)
    changepoint_err_list = []
    cluster_err_list= []
    iter_num =[]
    # loss_list = []
    for seed in range(M):
        ic_list = loss_mat_cp[seed, :] - Kl_mat[seed, :] + c_mat[seed,:]*C
        setpath(trans_setting, K = 3, init=init_list[np.argmax(ic_list)])
        file_name = "seed_"+str(seed)+".dat"
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                t['changepoint'].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        # loss_list.append(loss_mat_cp[seed, np.argmax(ic_list)])
        iter_num.append(t['iter_num'])
        pkl_file.close()
    res_diffC_cp[C_list.index(C)] = {"C":C,
                    "changepoint_err": np.mean(changepoint_err_list),
                    "ARI": np.mean(cluster_err_list),
                    # "loss":np.mean(loss_list),
                    "iter_num":np.mean(iter_num),
                    # "bestK":np.mean(bestK_list),
                    "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                    'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                    'iter_num_var':np.std(iter_num)/np.sqrt(M)
                    # 'bestK_var':np.std(bestK_list)/np.sqrt(M)
                    }

dat_cp = pd.DataFrame(res_diffC_cp)
dat_cp = round(dat_cp, 3)
for i in range(1,4):
    dat_cp.iloc[:,i] = [str(a) +'('+ str(b)+")" for a,b in zip(dat_cp.iloc[:,i],dat_cp.iloc[:,i+3])]
col = dat_cp.columns.tolist()
dat_cp = dat_cp[[col[0], col[4]]+ col[1:4]]
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
# file_name="results/2x3/tables/res0603.xlsx"
# dat_cp.to_excel(file_name, index=False,sheet_name = "cp_ic")  

# all performances
dat_all_cp = pd.DataFrame({"init":init_list,
              "changepoint_err":np.mean(changepoint_err_mat_cp, axis=0),
              "ARI":np.mean(cluster_err_mat_cp, axis=0),
              "iter_num":np.mean(iter_mat_cp, axis=0),
              "cp_var":np.std(changepoint_err_mat_cp, axis=0)/np.sqrt(M),
              "cluster_var":np.std(cluster_err_mat_cp, axis=0)/np.sqrt(M),
              "iter_num_var":np.std(iter_mat_cp,axis=0)/np.sqrt(M)})
dat_all_cp.iloc[:,1:7] = round(dat_all_cp.iloc[:,1:7], 3)
for i in range(1,4):
    dat_all_cp.iloc[:,i] = [str(a) +'('+ str(b)+")" for a,b in zip(dat_all_cp.iloc[:,i],dat_all_cp.iloc[:,i+3])]
col = dat_all_cp.columns.tolist()
dat_all_cp = dat_all_cp[col[:4]]
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
file_name="results/2x3/tables/res0603_2.xlsx"
# with pd.ExcelWriter(file_name) as writer:
#     dat_cp.to_excel(writer, sheet_name="IC", index=False)
#     dat_all_cp.to_excel(writer, sheet_name="all", index=False)
# with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#     dat_all_cp.to_excel(writer, sheet_name="cp all", index=False)

#%% analyse results 2 -- oracle
K = 3
oracle_list = ["changepoints_ora", "clustering_ora"]
res_oracle = [None] * len(oracle_list)
for init in oracle_list :
    ic_list = np.zeros(1)
    loss_mat = np.zeros([M, 1])
    Kl_mat = np.zeros([M, 1])
    c_mat = np.zeros([M, 1])
                
    bestK_list = np.zeros(M)
    changepoint_err_list = []
    cluster_err_list= []
    iter_num =[]
    for seed in range(M):
        setpath(trans_setting, K = K, init=init)
        file_name = "seed_"+str(seed)+".dat"
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        # print('seed',seed,t['iter_num'])
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                t['changepoint'].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        iter_num.append(t['iter_num'])
        pkl_file.close()
    res_oracle[oracle_list.index(init)] = {"oracle":init,
                                  "changepoint_err": np.mean(changepoint_err_list),
                                  "ARI": np.mean(cluster_err_list),
                                  "iter_num":np.mean(iter_num),
                                  "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                                  'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                                  'iter_num_var':np.std(iter_num)/np.sqrt(M)}


dat_oracle = pd.DataFrame(res_oracle)
dat_oracle = round(dat_oracle, 3)
for i in range(1,4):
    dat_oracle.iloc[:,i] = [str(a) +'('+ str(b)+")" for a,b in zip(dat_oracle.iloc[:,i],dat_oracle.iloc[:,i+3])]
col = dat_oracle.columns.tolist()
dat_oracle = dat_oracle[[col[0]]+ col[1:4]]
# os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
# #     dat_all_cp.to_excel(writer, sheet_name="all", index=False)
# with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
#     dat_oracle.to_excel(writer, sheet_name="oracle", index=False)
# # dat_oracle.to_excel(file_name)  
#%% analyse result 3 -- random cluster K_list all K's performance + consistent Kmeans
K_list = range(2, 6)
init_list= ["randomclustering", "clustering_Kmeans"]
## ic 
loss_mat_K = np.zeros([M, len(K_list)+1])
Kl_mat = np.zeros([M, len(K_list)+1])
c_mat = np.zeros([M, len(K_list)+1])
changepoint_err_mat_K = np.zeros([M,len(K_list)+1])
cluster_err_mat_K = np.zeros([M,len(K_list)+1])
iter_mat_K = np.zeros([M,len(K_list)+1])

for init in init_list:
    if init == "randomclustering":
        for K in K_list:
            for seed in range(M):
                for K in K_list:
                    setpath(trans_setting, K = K, init=init)
                    file_name = "seed_"+str(seed)+".dat"
                    pkl_file = open(file_name, 'rb')
                    t = pickle.load(pkl_file)
                    # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
                    #                               t['group'], N, T, K, C)
                    loss_mat_K[seed, K_list.index(K)] = t['loss']
                    Kl_mat[seed, K_list.index(K)] =  K*np.log(np.sum(T-1 -t['changepoint']))
                    Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
                    c_mat[seed, K_list.index(K)] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
                    changepoint_err_mat_K[seed,K_list.index(K)],cluster_err_mat_K[seed,K_list.index(K)] = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                            t['changepoint'].squeeze(), N, T)
                    iter_mat_K[seed,K_list.index(K)] = t['iter_num']
                    pkl_file.close()
    elif init == "clustering_Kmeans":
        for seed in range(M):
            setpath(trans_setting, K = 3, init=init)
            file_name = "seed_"+str(seed)+".dat"
            pkl_file = open(file_name, 'rb')
            t = pickle.load(pkl_file)
            # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
            #                               t['group'], N, T, K, C)
            loss_mat_K[seed, len(K_list)] = t['loss']
            Kl_mat[seed, len(K_list)] =  K*np.log(np.sum(T-1 -t['changepoint']))
            Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
            c_mat[seed, len(K_list)] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
            changepoint_err_mat_K[seed,len(K_list)],cluster_err_mat_K[seed,len(K_list)] = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                    t['changepoint'].squeeze(), N, T)
            iter_mat_K[seed,len(K_list)] = t['iter_num']
            pkl_file.close()

res_diffC_K = [None] * len(C_list)
for C in C_list:
    bestK_list = np.zeros(M)
    changepoint_err_list = []
    cluster_err_list= []
    iter_num =[]
    # loss_list = []
    for seed in range(M):
        ic_list = loss_mat_K[seed,:] - Kl_mat[seed,:] + c_mat[seed,:]*C
        if np.argmax(ic_list) == len(K_list):
            bestK = 3
            init = "clustering_Kmeans"
        else:
            bestK = K_list[np.argmax(ic_list)]
            init = "randomclustering"
        bestK_list[seed] = bestK
        setpath(trans_setting, K = bestK, init=init)
        file_name = "seed_"+str(seed)+".dat"
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                t['changepoint'].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        # loss_list.append(loss_mat_cp[seed, np.argmax(ic_list)])
        iter_num.append(t['iter_num'])
        pkl_file.close()
    res_diffC_K[C_list.index(C)] = {"C":C,
                                     "bestK":np.mean(bestK_list),
                    "changepoint_err": np.mean(changepoint_err_list),
                    "ARI": np.mean(cluster_err_list),
                    "iter_num":np.mean(iter_num),
                    # "bestK":np.mean(bestK_list),
                    'bestK_var':np.std(bestK_list)/np.sqrt(M),
                    "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                    'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                    'iter_num_var':np.std(iter_num)/np.sqrt(M)
                    }

dat_K = pd.DataFrame(res_diffC_K)
dat_K = round(dat_K, 3)
for i in range(1,5):
    dat_K.iloc[:,i] = [str(a) +'('+ str(b)+")" for a,b in zip(dat_K.iloc[:,i],dat_K.iloc[:,i+4])]
col = dat_K.columns.tolist()
dat_K = dat_K[col[:5]]

## all performance
dat_all_K = pd.DataFrame({"K":list(K_list) + [3],
                      "changepoint_err":np.mean(changepoint_err_mat_K, axis=0),
                      "ARI":np.mean(cluster_err_mat_K, axis=0),
                      "iter_num":np.mean(iter_mat_K, axis=0),
                      "cp_var":np.std(changepoint_err_mat_K, axis=0)/np.sqrt(M),
                      "cluter_var":np.std(cluster_err_mat_K, axis=0)/np.sqrt(M),
                      "iter_num_var":np.std(iter_mat_K,axis=0)/np.sqrt(M)})
# dat_K = pd.DataFrame(dat_K, columns=['K','changepoint_err', 'cluster_err',
#                                      'loss','iter_num','cp_var','cluster_var',
#                                      'loss_var','iter_num_var'])
dat_all_K = round(dat_all_K, 3)
for i in range(1,4):
    dat_all_K.iloc[:,i] = [str(a)+"("+str(b)+")" for a, b in zip(dat_all_K.iloc[:,i], dat_all_K.iloc[:,i+3])]

cols = dat_all_K.columns.tolist()
dat_all_K = dat_all_K[cols[:4]]
dat_all_K = dat_all_K.iloc[[4]+list(range(len(K_list)))]
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")

#%% analyse result 4 -- ic best K when the 2 kinds of initial estimators are combined
# C_list =[0,5,8,10,20, 50]
## ic
def IC_include_path_loss(init_list,C_list, N,T, M):
    include_path_loss = 1
    # init_list = ["changepoints_ora", "clustering_ora","changepoint_separ", "changepoint_no", "changepoints_random", 'randomclustering']
    # init_list = ["changepoints_ora", 'randomclustering']
    C_list = np.arange(0,10,1).tolist()
    K_list = range(2, 6)
    # ic_list = [None] * len(init_list)
    loss_mat = [None] * len(init_list)
    ic_mat = [None] * len(init_list)
    Kl_mat = [None] * len(init_list)
    c_mat = [None] * len(init_list)
    for C in C_list:
        init_count = np.zeros(len(init_list))
        K_count = np.zeros(len(K_list))
        bestK_list = []
        changepoint_err_list = []
        cluster_err_list= []
        iter_num =[]
        for seed in range(M):
            best_init = None
            bestK = None
            bestIC = None
            bestiter = None
            States, Rewards, Actions, changepoints_true,g_index_true = gen_dat(N=N, T=T, K=3, coef=coef, 
                                                                  signal =signal,changepoint_list=None,
                                                                  trans_setting=trans_setting,seed=seed + 100)

            for init in init_list:
                if init == "randomclustering":
                    loss_mat[init_list.index(init)] = np.zeros([M, 1])
                    ic_mat[init_list.index(init)] = np.zeros([M, 1])
                    Kl_mat[init_list.index(init)] = np.zeros([M, 1])
                    c_mat[init_list.index(init)] = np.zeros([M, 1])
                    ic_randomK_eachseed = []
                    iter_num = []
                    for K in K_list:
                        setpath(trans_setting, K = K, init=init)
                        file_name = "seed_"+str(seed)+".dat"
                        pkl_file = open(file_name, 'rb')
                        t = pickle.load(pkl_file)
                        pkl_file.close()
                        loss_tmp = []
                        Kl_tmp = []
                        c_mat_tmp = []
                        cp_rev = t['changepoint_eachiter'][:, range(t['changepoint_eachiter'].shape[1]-1,-1,-1)]
                        g_rev = t['g_index_eachiter'][:, range(t['g_index_eachiter'].shape[1]-1,-1,-1)]
                        for ii in range(min(t['changepoint_eachiter'].shape[1], t['g_index_eachiter'].shape[1])):
                            loss_tmp.append(mean_detect.goodnessofClustering(States, N, T, K, cp_rev[:, ii], Actions, g_rev[:,ii]))
                        loss_tmp = loss_tmp [::-1]
                        Kl_tmp, c_mat_tmp = uti.paramInIC(t, N, K, T, include_path_loss)
                        ic_tmp =  np.array(loss_tmp) - np.array(Kl_tmp) +  np.array(c_mat_tmp) * C
                        iter_num.append(np.argmax(ic_tmp))
                        ic_randomK_eachseed.append(np.max(ic_tmp))
                    if bestIC is None:
                        best_init = init
                        bestIC = np.max(ic_randomK_eachseed)
                        bestK = K_list[np.argmax(ic_randomK_eachseed)]
                        if bestK == 2:
                            print('seed', seed)
                        bestiter = iter_num[np.argmax(ic_randomK_eachseed)]
                    elif bestIC < np.max(ic_randomK_eachseed):
                        best_init = init
                        bestIC = np.max(ic_randomK_eachseed)
                        bestK = K_list[np.argmax(ic_randomK_eachseed)]
                        bestiter = iter_num[np.argmax(ic_randomK_eachseed)]
                else:
                    loss_mat[init_list.index(init)] = np.zeros([M, 1])
                    Kl_mat[init_list.index(init)] = np.zeros([M, 1])
                    c_mat[init_list.index(init)] = np.zeros([M, 1])
                    setpath(trans_setting, K = 3, init=init)
                    file_name = "seed_"+str(seed)+".dat"
                    pkl_file = open(file_name, 'rb')
                    t = pickle.load(pkl_file)
                    pkl_file.close()
                    loss_tmp = []
                    Kl_tmp = []
                    c_mat_tmp = []
                    cp_rev = t['changepoint_eachiter'][:, range(t['changepoint_eachiter'].shape[1]-1,-1,-1)]
                    g_rev = t['g_index_eachiter'][:, range(t['g_index_eachiter'].shape[1]-1,-1,-1)]
                    for ii in range(min(t['changepoint_eachiter'].shape[1], t['g_index_eachiter'].shape[1])):
                        if len(set(g_rev[:,ii])) == 1:
                            loss_tmp.append(-sys.float_info.max)
                        else:
                            loss_tmp.append(mean_detect.goodnessofClustering(States, N, T, 3, cp_rev[:, ii], Actions, g_rev[:,ii]))
                    loss_tmp = loss_tmp [::-1]
                    Kl_tmp, c_mat_tmp = uti.paramInIC(t, N, 3, T, include_path_loss =0)
                    ic_tmp =  np.array(loss_tmp) - np.array(Kl_tmp) +  np.array(c_mat_tmp) * C
                    # ic_mat[init_list.index(init)][seed, 0] = np.max(ic_tmp)
                    if bestIC is None:
                        best_init = init
                        bestIC = np.max(ic_tmp)
                        bestK = 3
                        bestiter = np.argmax(ic_tmp)
                    elif bestIC < np.max(ic_tmp):
                        best_init = init
                        bestIC = np.max(ic_tmp)
                        bestK = 3
                        bestiter = np.argmax(ic_tmp)
                  
            # collect the best performance for this seed
            print('K', bestK, 'init', best_init)
            setpath(trans_setting, K = bestK, init=best_init)
            file_name = "seed_"+str(seed)+".dat"
            pkl_file = open(file_name, 'rb')
            t = pickle.load(pkl_file)
            bestK_list.append(bestK)
            # print(t['changepoint'])
            if t['changepoint_eachiter'].shape[1] == t['g_index_eachiter'].shape[1]:
                cp = t['changepoint_eachiter'][:, bestiter]
                g_index = t['g_index_eachiter'][:, bestiter]
            else:
                cp = t['changepoint_eachiter'][:, bestiter+1]
                g_index = t['g_index_eachiter'][:, bestiter]
            changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), g_index.squeeze(),
                                                    cp.squeeze(), N, T)
            changepoint_err_list.append(changepoint_err)
            # print('cp_err_best', changepoint_err, "ARI", cluster_err)
            cluster_err_list.append(cluster_err)
            iter_num.append(bestiter)
            pkl_file.close()
            init_count[init_list.index(best_init)] = init_count[init_list.index(best_init)] + 1
            K_count[K_list.index(bestK)] = K_count[K_list.index(bestK)] +1
        res_diffC_all[C_list.index(C)] = {"C":C,
                                      "changepoint_err": np.mean(changepoint_err_list),
                                      "ARI": np.mean(cluster_err_list),
                                      "iter_num":np.mean(iter_num),
                                      "bestK":np.mean(bestK_list),
                                      "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                                      'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                                      'iter_num_var':np.std(iter_num)/np.sqrt(M),
                                      'bestK_var':np.std(bestK_list)/np.sqrt(M),
                                      "init_count":init_count,
                                      "K_count":K_count}        
    dat_all = pd.DataFrame(res_diffC_all)
    dat_all = round(dat_all, 3)
    for i in range(1,5):
        dat_all.iloc[:,i] = [str(a) +'('+ str(b)+")" 
                             for a,b in zip(dat_all.iloc[:,i],dat_all.iloc[:,i+4])]
    col = dat_all.columns.tolist()
    dat_all = dat_all[[col[0], col[4]]+ col[1:4]]
    print(dat_all)
    return dat_all, res_diffC_all


res_diffC_all = [None] * len(C_list)
for C in C_list:
    init_count = np.zeros(len(init_list))
    K_count = np.zeros(len(K_list))
    bestK_list = np.zeros(M)
    changepoint_err_list = []
    cluster_err_list= []
    iter_num =[]
    for seed in range(M):
        # print('seed', seed)
        for init in init_list:
            # print('====init:', init,"====")
            if init == "randomclustering":
                ic_list[init_list.index(init)] = loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C
                clusterK = K_list[np.argmax(ic_list[init_list.index('randomclustering')])]
                # print('ic_list[init_list.index("randomclustering")]',ic_list[init_list.index('randomclustering')])
                ic_list[init_list.index('randomclustering')] = np.max(ic_list[init_list.index('randomclustering')])
            else:    
                # print('loss_mat[',init_list.index(init),'][seed, :]',loss_mat[init_list.index(init)][seed, :], 'Kl', Kl_mat[init_list.index(init)][seed, :] ,'cmat', c_mat[init_list.index(init)][seed,:]*C)
                ic_list[init_list.index(init)] = (loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C)[0]
        # print('ic_list' ,ic_list)
        init = init_list[np.where(ic_list == np.max(ic_list))[0][0]]
        init_count[np.where(ic_list == np.max(ic_list))[0][0]] = init_count[np.where(ic_list == np.max(ic_list))[0][0]] + 1
        if init == "randomclustering":
            bestK = clusterK
        else:
            bestK = 3
        K_count[K_list.index(bestK)] = K_count[K_list.index(bestK)] +1
        # print('**** init', init, 'K', bestK)
        # bestK = K_list[np.where(ic_list == np.max(ic_list))[1][0]]
        bestK_list[seed]=bestK
        setpath(trans_setting, K = bestK, init=init)
        file_name = "seed_"+str(seed)+".dat"
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        # print(t['changepoint'])
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                t['changepoint'].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        # print('cp_err_best', changepoint_err, "ARI", cluster_err)
        cluster_err_list.append(cluster_err)
        iter_num.append(t['iter_num'])
        pkl_file.close()
        
    res_diffC_all[C_list.index(C)] = {"C":C,
                                  "changepoint_err": np.mean(changepoint_err_list),
                                  "ARI": np.mean(cluster_err_list),
                                  "iter_num":np.mean(iter_num),
                                  "bestK":np.mean(bestK_list),
                                  "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                                  'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                                  'iter_num_var':np.std(iter_num)/np.sqrt(M),
                                  'bestK_var':np.std(bestK_list)/np.sqrt(M),
                                  "init_count":init_count,
                                  "K_count":K_count}

dat_all = pd.DataFrame(res_diffC_all)
dat_all = round(dat_all, 3)
for i in range(1,5):
    dat_all.iloc[:,i] = [str(a) +'('+ str(b)+")" 
                         for a,b in zip(dat_all.iloc[:,i],dat_all.iloc[:,i+4])]
col = dat_all.columns.tolist()
dat_all = dat_all[[col[0], col[4]]+ col[1:4]]
print(dat_all)
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
file_name="results/2x3/tables/res_0609_path"+".xlsx"
dat_all.to_excel(file_name)  

init_list = ["changepoint_separ", "changepoint_no", "changepoints_random", 'randomclustering']
C_list = np.arange(0,10,1).tolist()
dat_all_noora_path, res_tmp = IC_include_path_loss(init_list, C_list, N, T, M)
with pd.ExcelWriter(file_name, mode="a", engine="openpyxl") as writer:
    dat_all_noora_path.to_excel(writer, sheet_name="no ora", index=False)

#%% analyse result 5 -- ic best K no oracle
# C_list =[0, 0.1, 1, 10,50,100,200,300,400,500, 1000]
## ic
init_list = ["changepoint_separ", "changepoint_no", "changepoints_random", 'randomclustering']

K_list = range(2, 6)
ic_list = [None] * len(init_list)
loss_mat = [None] * len(init_list)
Kl_mat = [None] * len(init_list)
c_mat = [None] * len(init_list)
for init in init_list:
    if init == "randomclustering":
        loss_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
        Kl_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
        c_mat[init_list.index(init)] = np.zeros([M, len(K_list)])
        for seed in range(M):
            for K in K_list:
                setpath(trans_setting, K = K, init=init)
                file_name = "seed_"+str(seed)+".dat"
                pkl_file = open(file_name, 'rb')
                t = pickle.load(pkl_file)
                # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
                #                               t['group'], N, T, K, C)
                loss_mat[init_list.index(init)][seed, K_list.index(K)] = t['loss']# / np.mean(T - t['changepoint'] - 1)
                # Kl_mat[init_list.index(init)][seed, K_list.index(K)] =  K*np.log(np.sum(T-1 -t['changepoint']))
                # Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
                # c_mat[init_list.index(init)][seed, K_list.index(K)] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
                pkl_file.close()
                Kl_mat[init_list.index(init)][seed, K_list.index(K)], c_mat[init_list.index(init)][seed, K_list.index(K)]=uti.paramInIC(t, N, K, T)
    else:
        loss_mat[init_list.index(init)] = np.zeros([M, 1])
        Kl_mat[init_list.index(init)] = np.zeros([M, 1])
        c_mat[init_list.index(init)] = np.zeros([M, 1])
        for seed in range(M):
            setpath(trans_setting, K = 3, init=init)
            file_name = "seed_"+str(seed)+".dat"
            pkl_file = open(file_name, 'rb')
            t = pickle.load(pkl_file)
            # ic_list[K_list.index(K)] = uti.IC(t['loss'], t['changepoint'], 
            #                               t['group'], N, T, K, C)
            # loss = mean_detect.goodnessofClustering(States, N, T, 3, t['changepoint'], Actions, t['group'])
            loss_mat[init_list.index(init)][seed, 0] = t['loss'] #/ np.mean(T - t['changepoint'] - 1)
            # Kl_mat[init_list.index(init)][seed, 0] =  K*np.log(np.sum(T-1 -t['changepoint']))
            # Ck, indicesList, occurCount = np.unique(t['group'], return_index = True,return_counts=True)
            # c_mat[init_list.index(init)][seed, 0] = occurCount.dot((T-1 -t['changepoint'])[np.s_[indicesList]])/(N*T)*np.log(np.sum(T-1 -t['changepoint']))
            pkl_file.close()
            Kl_mat[init_list.index(init)][seed, 0],c_mat[init_list.index(init)][seed, 0] = uti.paramInIC(t, N, 3, T)


# C_list = [0,10,100,200,400]
res_diffC_noora = [None] * len(C_list)
for C in C_list:
    bestK_list = np.zeros(M)
    changepoint_err_list = []
    cluster_err_list= []
    iter_num =[]
    for seed in range(M):
        for init in init_list:
            if init == "randomclustering":
                ic_list[init_list.index(init)] = loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C
                clusterK = K_list[np.argmax(ic_list[init_list.index('randomclustering')])]
                ic_list[init_list.index('randomclustering')] = np.max(ic_list[init_list.index('randomclustering')])
            else:    
                ic_list[init_list.index(init)] = (loss_mat[init_list.index(init)][seed, :] - Kl_mat[init_list.index(init)][seed, :] + c_mat[init_list.index(init)][seed,:]*C)[0]
        init = init_list[np.where(ic_list == np.max(ic_list))[0][0]]
        if init == "randomclustering":
            bestK = clusterK
        else:
            bestK = 3
        # bestK = K_list[np.where(ic_list == np.max(ic_list))[1][0]]
        bestK_list[seed]=bestK
        setpath(trans_setting, K = bestK, init=init)
        file_name = "seed_"+str(seed)+".dat"
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        changepoint_err, cluster_err = evaluate(changepoints_true.squeeze(), t['group'].squeeze(),
                                                t['changepoint'].squeeze(), N, T)
        changepoint_err_list.append(changepoint_err)
        cluster_err_list.append(cluster_err)
        iter_num.append(t['iter_num'])
        pkl_file.close()
    res_diffC_noora[C_list.index(C)] = {"C":C,
                                  "changepoint_err": np.mean(changepoint_err_list),
                                  "ARI": np.mean(cluster_err_list),
                                  "iter_num":np.mean(iter_num),
                                  "bestK":np.mean(bestK_list),
                                  "cp_var":np.std(changepoint_err_list)/np.sqrt(M),
                                  'cluster_var':np.std(cluster_err_list)/np.sqrt(M),
                                  'iter_num_var':np.std(iter_num)/np.sqrt(M),
                                  'bestK_var':np.std(bestK_list)/np.sqrt(M)}

dat_all_noora = pd.DataFrame(res_diffC_noora)
dat_all_noora = round(dat_all_noora, 3)
for i in range(1,5):
    dat_all_noora.iloc[:,i] = [str(a) +'('+ str(b)+")" 
                         for a,b in zip(dat_all_noora.iloc[:,i],dat_all_noora.iloc[:,i+4])]
col = dat_all_noora.columns.tolist()
dat_all_noora = dat_all_noora[[col[0], col[4]]+ col[1:4]]
print(dat_all_noora)
# file_name="results/2x3/res_combined_2_init_estimators"+".xlsx"
# dat_all.to_excel(file_name)  

#%% write to excel
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
#%% write to excel
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu")
file_name="results/2x3/tables/res0609.xlsx"
with pd.ExcelWriter(file_name) as writer:
    dat_cp.to_excel(writer, sheet_name="cp ic", index=False)
    dat_all_cp.to_excel(writer, sheet_name="cp all", index=False)
    dat_oracle.to_excel(writer, sheet_name = "oracle", index=False)
    dat_K.to_excel(writer, sheet_name = "K ic", index=False)
    dat_all_K.to_excel(writer, sheet_name = "K all", index=False)
    dat_all.to_excel(writer, sheet_name = "all ic", index=False)
    dat_all_noora.to_excel(writer, sheet_name = "no ora ic", index=False)