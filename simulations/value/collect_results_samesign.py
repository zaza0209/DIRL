# collect the results
import pickle, os, sys, re
import numpy as np
import pandas as pd
from datetime import date
sys.path.append('/home/huly0209_gmail_com/heterRL')
print(sys.path)
import simu.utilities as ut
is_save = int(sys.argv[1])
M=20
Ns = [50]
T_new = 250
cp_detect_interval = 25
seeds = range(M)
cov=0.25
C=5
gamma = 0.9
effect_size ="samesigstrong" # "strong""0.4""moderate"
trans_setting = ['pwconst2','smooth']#,]
# inits = ['true_clustering', "true_change_points",'kmeans',"random_clustering" ]
type_all = ['proposed', 
            'oracle','only_cp', 'only_clusters', 
'overall'
]
type_list = [type_all#,
         # [ 'true_clustering', "no_clusters"],
         # ["true_change_points", "no_change_points"],
         # ["true_change_points", "random_cp_K2"],
         # ['true_clustering',"random_clustering"],
         # ["random_cp_K1","random_cp_K2","random_cp_K3","random_cp_K4",'kmeans_K1', 'kmeans_K2', 'kmeans_K3', 'kmeans_K4']
         ]
file_name_index = ['all','(14)','(23)','(25)', '(16)','(7)']
today = date.today()

path_original = os.getcwd()

def run(type_all):
    #%% ##### error #######
    os.chdir(path_original+ "/results/")
    print('os.getcwd()',os.getcwd())
    results_list = [[None] * len(Ns) for i in range(len(trans_setting))]
    res_all = []
    for setting in trans_setting:
        for N in Ns:
            print('N',N)
            res_mat = np.zeros([len(type_all), 4])
            for type_est in type_all:
                print('type_est',type_est)
                # if init != "random_cp":
                av_r = []
                dis_r = []
                for seed in seeds:
                    os.chdir(path_original+ "/results/")
                    print('seed',seed)
                    path_name = './trans' + setting +'/effect_size_samesign0.4'+'/N' + str(N) +'/Tnew_' +\
                                str(T_new)+'_type_'+type_est+'_cpitv'+ str(cp_detect_interval)+'/cov'+str(cov) + '/seed'+str(seed) 
# 	                print('path_name', path_name )
                    print(path_name)
                    if os.path.exists(path_name):
                        print('path_name', path_name )
                        os.chdir(path_name)	    	           
                        file_name = "seed_"+str(seed)+".dat"
                        if not os.path.exists(file_name):
                            print('=========== \n BUG! \n============')
                        else:
                            pkl_file = open(file_name, 'rb')
                            t = pickle.load(pkl_file)
                            av_r.append(t['value']['average_reward'])
                            dis_r.append(t['value']['discounted_reward'])
                            print("t['average_reward']",t['value']['average_reward'])
                            # file_name_new = 'seed_'+str(seed)+'.dat'
                            # with open(file_name_new, "wb") as f:
                            #     pickle.dump(t, f)
                            tmp = [setting, seed, type_est, t['value']['average_reward'],t['value']['discounted_reward']]
                            res_all.append(tmp)
                            pkl_file.close()
                    res_mat[type_all.index(type_est), 0]= np.mean(av_r) 
                    res_mat[type_all.index(type_est), 1] = np.mean(dis_r) 
                    res_mat[type_all.index(type_est), 2] = str(np.std(av_r)/np.sqrt(len(seeds)))
                    res_mat[type_all.index(type_est), 3] = str(np.std(dis_r)/np.sqrt(len(seeds)))
        
            print('res_mat',res_mat,'\n results_list',results_list)
            results_list[trans_setting.index(setting)][Ns.index(N)] = res_mat.copy()
                # print('results_list',results_list,'\n res_mat',res_mat)
    # print('results_list',results_list)
    res_all = np.array(res_all).reshape([-1, 5])
    res_all = pd.DataFrame(res_all)
    if is_save:
        os.chdir(path_original+ "/results/")
        if not os.path.exists('value'):
            os.makedirs('value', exist_ok=True)
        file_name="value/vall_"+str(today)+'N' + str(N) + "_1d.csv"          
        res_all.to_csv(file_name, index=False, header=['Setting','seed','init','Average Value', 'Discounted Value'])
 
    # for j in range(len(trans_setting)):        
    #     for i in range(len(Ns)):
    #         results_list[j][i] = pd.DataFrame(results_list[j][i])
    #         results_list[j][i] = round(results_list[j][i], 3)
    #         print('results_list[',j,'][',i,']',results_list[j][i])
    #         for z in range(2):
    #             results_list[j][i].iloc[:,z] = [str(a) +'('+ str(b)+")" 
    #                                  for a,b in zip(results_list[j][i].iloc[:,z],results_list[j][i].iloc[:,z+2])]
    #         col = results_list[j][i].columns.tolist()
    #         results_list[j][i] = results_list[j][i][[col[0],col[1]]]
    #         results_list[j][i].insert(loc=0, column="type", value = type_all)
    #         results_list[j][i].columns = ['init', 'av_r','dis_r']
    
    # if is_save:
    #     os.chdir(path_original+ "/results/")
    #     if not os.path.exists('value'):
    #         os.makedirs('value', exist_ok=True)
    #     file_name="value/value_"+str(today)+"_trans"+str(trans_setting) +'N' + str(Ns) + "_1d.xlsx"
    #     with pd.ExcelWriter(file_name) as writer:
    #          for setting in trans_setting:
    #              for N in Ns:
    #                   results_list[trans_setting.index(setting)][Ns.index(N)].to_excel(writer, sheet_name=setting + str(N), index=False)

   
#%%
run(type_all)
