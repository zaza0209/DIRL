# -*- coding: utf-8 -*-
"""
collect results

@author: test
"""
# collect the results
import platform, sys, os, pickle, re
plat = platform.platform()
print(plat)
if plat == 'Windows-10-10.0.14393-SP0': ##local
    os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/simu/tuneK_iterations/final_perf")
    sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
elif plat == 'Linux-5.10.0-18-cloud-amd64-x86_64-with-glibc2.31':  # biostat cluster
    os.chdir("/home/xx/heterRL/tuneK_iterations/final_perf")
    sys.path.append("/home/xx/heterRL")
else:
    os.chdir("/home/xx/heterRL/tuneK_iterations/final_perf")
    sys.path.append("/home/xx/heterRL")
import numpy as np
import pandas as pd
from datetime import date
# sys.path.append('/home/xx/heterRL')
# print(sys.path)
import functions.utilities as ut
calculate_err = int(sys.argv[1])
calculate_ic = int(sys.argv[2])
is_save_all = int(sys.argv[3])
is_save = int(sys.argv[4])
C=0
threshold_type = 'maxcusum'
M=20
Ns = [50]
N = 50
T = 50
seeds = range(M)
cov=0.25
#C=1
trans_setting = ['pwconst2']#,],'smooth'#
effect_size =" "#"strong"
# inits = ['true_clustering', "true_change_points",'kmeans',"random_clustering" ]
init_all = ['true_clustering', 
         "true_change_points", 
 "no_change_points" , 
  "no_clusters_K2",
   "random_cp_K2",
  # "random_cp_K2","random_cp_K3","random_cp_K4"#,
           "random_clustering" ,
     'kmeans_K1', 
     'kmeans_K2', 'kmeans_K3', 'kmeans_K4'
#"tuneK_iter_K2"
]
# inits = ["true_change_points", "no_change_points"]  
# inits = [ 'true_clustering', "no_clusters"]
# inits = ["true_change_points",  "random_cp"]
#           "random_clustering" ,
#     'kmeans'
# ]
init_list = [
     # init_all,
          # [ 'true_clustering', "no_clusters"],
          ["true_change_points", "no_change_points"],
          ["true_change_points", "random_cp_K2"],
          ["true_change_points","random_cp_K2", "no_change_points"],
          # ['true_clustering',"random_clustering"],
          ['kmeans_K1', 'kmeans_K2', 'kmeans_K3', 'kmeans_K4']
         ]
file_name_index = [#'all',
                   #'(14)',
                   '(23)','(25)', '(235)', #'(16)',
'(7)']
inits =['tuneK_iter']
random_cp_index = range(1, 5)
today = date.today()

path_original = os.getcwd()

def run(calculate_err, calculate_ic,is_save_all, inits, N=N):
    if is_save_all:
        os.chdir(path_original+ "/results/")
        print('os.getcwd()',os.getcwd())
        res_list = []
        for setting in trans_setting:
            for init in inits:
                print('init',init)
                cp_err = []
                ARI = []
                tmp = []
                for seed in seeds:
                    os.chdir(path_original+ "/results/")
                    print('seed',seed)
                    path_name = 'transpwconst2/effect_size_strong/N50_T50/K[1, 2, 3, 4]/init_tuneK_iter/seed'+str(seed)                     
                        # C:/Users/test/Dropbox/DIRL/IHS/simu/simu_original/tuneK_iterations/results/transpwconst2/effect_size_strong/N50_T50/K[1, 2, 3, 4]/init_tuneK_iter/seed1
                    print('path_name',path_name)
                    if os.path.exists(path_name):
                        os.chdir(path_name)
                        file_name = "seed_"+str(seed)+".dat"
                        pkl_file = open(file_name, 'rb')
                        t = pickle.load(pkl_file)
                        # cp_err.append(t['cp_err'])
                        print('t[cp_err]',t['cp_err'], ', ARI:', t['ARI'])
                        # if init == 'tuneK_iter':
                        print('t Kpath', t['K_path'])
                        # ARI.append(t['ARI'])
                        tmp = [setting, seed, 'best_model', t['cp_err'],t['ARI']]
                        res_list.append(tmp)
                        pkl_file.close()
                           
        res_list = np.array(res_list).reshape([-1, 5])
        res_list = pd.DataFrame(res_list)
        print('res_list',res_list)
        if is_save:
            os.chdir(path_original+ "/results/")
            # for setting in trans_setting:
            file_name="tuneK_iter_"+str(today)+'N' + str(N) + "_1d.csv"          
            res_list.to_csv(file_name, index=False, header=['Setting','seed','init','cp_err', 'ARI'])
     

    #%% ##### error #######
    if calculate_err :
        os.chdir(path_original+ "/results/")
        print('os.getcwd()',os.getcwd())
        results_list = [[None] * len(Ns) for i in range(len(trans_setting))]
        for setting in trans_setting:
            for N in Ns:
                print('N',N)
                res_mat = np.zeros([len(inits), 4])
                for init in inits:
                    print('init',init)
                    # if init != "random_cp":
                    cp_err = []
                    ARI = []
                    for seed in seeds:
                        os.chdir(path_original+ "/results/")
                        print('seed',seed)
                        path_name = './strong_threshold_'+ threshold_type+'/sim_result_trans' + setting +'/effect_size_samesign_sys0.25'+'/N' + str(N) +'/T_' + str(T)+'_init_'+init+\
                            '_1d/cov'+str(cov) + '/seed'+str(seed)
                        print('path_name',path_name)
                        if os.path.exists(path_name):
                            
                            os.chdir(path_name)
                            if re.search('random_cp', init):
                                file_name = "seed_"+str(seed)+"_ind0"+".dat"
                            else:
                                file_name = "seed_"+str(seed)+".dat"
                            if os.path.exists(file_name):
                                pkl_file = open(file_name, 'rb')
                                t = pickle.load(pkl_file)
                                cp_err.append(t['cp_err'])
                                print('t[cp_err]',t['cp_err'], ', ARI:', t['ARI'])
                                ARI.append(t['ARI'])
                                pkl_file.close()
                    print('cp_err', cp_err, ', ARI' ,ARI)
                    res_mat[inits.index(init), 0]= np.mean(cp_err) 
                    res_mat[inits.index(init), 1] = np.mean(ARI) 
                    res_mat[inits.index(init), 2] = str(np.std(cp_err)/np.sqrt(len(seeds)))
                    res_mat[inits.index(init), 3] = str(np.std(ARI)/np.sqrt(len(seeds)))
            
                print('res_mat',res_mat,'\n results_list',results_list)
                results_list[trans_setting.index(setting)][Ns.index(N)] = res_mat.copy()
                    # print('results_list',results_list,'\n res_mat',res_mat)
        # print('results_list',results_list)
        for j in range(len(trans_setting)):        
            for i in range(len(Ns)):
                results_list[j][i] = pd.DataFrame(results_list[j][i])
                results_list[j][i] = round(results_list[j][i], 3)
                print('results_list[',j,'][',i,']',results_list[j][i])
                for z in range(2):
                    results_list[j][i].iloc[:,z] = [str(a) +'('+ str(b)+")" 
                                         for a,b in zip(results_list[j][i].iloc[:,z],results_list[j][i].iloc[:,z+2])]
                col = results_list[j][i].columns.tolist()
                results_list[j][i] = results_list[j][i][[col[0],col[1]]]
                results_list[j][i].insert(loc=0, column="init", value = inits)
                results_list[j][i].columns = ['init', 'cp_err','ARI']
        
        if is_save:
            os.chdir(path_original+ "/results/error_res")
            file_name="error"+str(today)+"_trans"+str(trans_setting) +'N' + str(Ns) + "_1d.xlsx"
            with pd.ExcelWriter(file_name) as writer:
                 for setting in trans_setting:
                     for N in Ns:
                          results_list[trans_setting.index(setting)][Ns.index(N)].to_excel(writer, sheet_name=setting + str(N), index=False)
    
    #%% ic
    if calculate_ic:
        error_file = [[None] * len(Ns) for i in range(len(trans_setting))]
        init_file = [[None] * len(Ns) for i in range(len(trans_setting))]
        res_list = []
        for setting in trans_setting:
            print('setting', setting)
            for N in Ns:
                cp_err = []
                ARI_all = []
                best_init = []
                for seed in range(M):
                    print('*********** seed', seed, '***********')
                    IC_max = None
                    for init in inits:
                        print('----- init', init)
                        os.chdir(path_original+ "/results/")
                        path_name = './strong_threshold_'+ threshold_type+'/sim_result_trans' + setting +'/N' + str(N) +'/T_' + str(T)+'_init_'+init+\
                            '_1d/cov'+str(cov) + '/seed'+str(seed)
                        if os.path.exists(path_name):
                            os.chdir(path_name)
                            # print(path_name)
                            if re.search("kmeans", init):
                                K  = int(''.join([i for i in init if i.isdigit()]))
                                file_name = "seed_"+str(seed)+".dat"
                                pkl_file = open(file_name, 'rb')
                                t = pickle.load(pkl_file)
                                tmp = [setting, seed, init, t['cp_err'], t['ARI']]
                                res_list.append(tmp)#/np.mean(T-1-t['changepoints'])
                                IC_tmp = ut.IC(t['loss'], t['changepoints'], t['clustering'], N, T,K=K, C=C, Kl_fun = 'logN')
                                print('loglikelihood loss', t['loss']/np.mean(T-1-t['changepoints']), ', IC', IC_tmp,', cp', t['cp_err'], ', ARI', t['ARI'], ', tau bar: ',np.mean(T-1-t['changepoints']))
                                if IC_max is None:
                                    print('MAX!!!')
                                    IC_max = IC_tmp
                                    best_model = init
                                    cp = t['cp_err']
                                    ARI = t['ARI']
                                elif IC_max < IC_tmp:
                                    print('MAX!!!')
                                    IC_max = IC_tmp
                                    best_model = init
                                    cp = t['cp_err']
                                    ARI = t['ARI']
                                pkl_file.close()                            
                            else:
                                file_name = "seed_"+str(seed)+".dat"
                                pkl_file = open(file_name, 'rb')
                                t = pickle.load(pkl_file)
                                tmp = [setting, seed, init, t['cp_err'], t['ARI']]
                                res_list.append(tmp)
                                IC_tmp = ut.IC(t['loss'], t['changepoints'], t['clustering'], N, T,K=2, C=C, Kl_fun = 'logN')
                                print(' loss', t['loss']/np.mean(T-1-t['changepoints']), ', IC', IC_tmp,', cp', t['cp_err'], ', ARI', t['ARI'])
                                if IC_max is None:
                                    print('MAX!!!')
                                    IC_max = IC_tmp
                                    best_model = init
                                    cp = t['cp_err']
                                    ARI = t['ARI']
                                elif IC_max < IC_tmp:
                                    print('MAX!!!')
                                    IC_max = IC_tmp
                                    best_model = init
                                    cp = t['cp_err']
                                    ARI = t['ARI']
                                pkl_file.close()
                            
                    cp_err.append(cp)
                    ARI_all.append(ARI)
                    best_init.append(best_model)
                    print('bestinit',best_model, ', cp', cp, ', ARI', ARI)

                    tmp = [setting, seed,'best_model', cp, ARI]
                    res_list.append(tmp)
                error_mat= np.zeros([2,2])
                print('cp_all', cp_err, 'ARI', ARI_all)
                error_mat[0,0] = round(np.mean(cp_err),3)
                error_mat[0,1] = round(np.mean(ARI_all),3)
                error_mat[1,0] = round(np.std(cp_err)/np.sqrt(M), 3)
                error_mat[1,1] = round(np.std(ARI_all)/np.sqrt(M), 3)
                error_mat= pd.DataFrame(error_mat)
                error_mat.iloc[0,:] = [str(a) +'('+ str(b)+")" 
                                     for a,b in zip(error_mat.iloc[0,:],error_mat.iloc[1,:])]
    
                error_file[trans_setting.index(setting)][Ns.index(N)]=error_mat.iloc[0,:]
                print('error_file', error_file)
                init_chosen, occurCount = np.unique(best_init, return_index = False,return_counts=True)
                init_mat = pd.DataFrame({'init':init_chosen, 'count':occurCount})
                init_file[trans_setting.index(setting)][Ns.index(N)]=init_mat
                print('init_file', init_file)
                
        if is_save:
            # os.chdir(path_original+ "/results/ic_res")
            # file_name="ic"+str(today)+"_trans"+str(trans_setting) +'N' + str(Ns)+file_name_index[init_list.index(inits)]+ "_1d.xlsx"
            # with pd.ExcelWriter(file_name) as writer:
            #     for setting in trans_setting:
            #         for N in Ns:
            #             error_file[trans_setting.index(setting)][Ns.index(N)].to_excel(writer, sheet_name=setting + str(N), index=False)
            #             # startrow = writer.sheets[setting + str(N)].max_row + 2
            #             init_file[trans_setting.index(setting)][Ns.index(N)].to_excel(writer,  sheet_name=setting + str(N), startrow = 5, index = False, header = False)
    
            res_list = np.array(res_list).reshape([-1, 5])
            res_list = pd.DataFrame(res_list)
            os.chdir(path_original+ "/results/")
            # for setting in trans_setting:
            file_name="icmodel_"+str(today)+'method'+file_name_index[init_list.index(inits)]+'N' + str(N) + "_1d.csv"          
            res_list.to_csv(file_name, index=False, header=['Setting','seed','init','cp_err', 'ARI'])
     


    return
#%%
if calculate_err:
    run(calculate_err, 0, 0, inits)
if calculate_ic:
    for inits in init_list:
        run(0, 1,0, inits)
if is_save_all:
    run(0, 0, 1, init_list[0])
	