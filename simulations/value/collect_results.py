##############################################################
#### collect data and save it in csv for R displaying ####
##############################################################
import pickle, os, sys, re
import numpy as np
import pandas as pd
from datetime import date
# sys.path.append('/home/huly0209_gmail_com/heterRL')
# print(sys.path)
is_save = int(sys.argv[1])
max_iter=int(sys.argv[2])
M=20
N = 50
T_new = 250
cp_detect_interval = 25
seeds = range(M)
cov=0.25
gamma = 0.9
effect_size = "strong"
trans_setting = ['pwconst2','smooth'] 
type_all = ['proposed', 'oracle','only_cp', 'only_clusters', 'overall']
today = date.today()
path_original = os.getcwd()
Kl_fun='Nlog(NT)/T' #"sqrtN"
K_list=[1,2,3,4]
early_stopping=0
C=1
#################
def run_collect(is_save):
    os.chdir(path_original+ "/results/")
    print('os.getcwd()',os.getcwd())
    res_list = []
    for setting in trans_setting:
        for type_est in type_all:
            if type_est in ["proposed", "only_clusters"]:
                if Kl_fun=='Nlog(NT)/T':
                    Kl_fun_name = "Nlog(NT)_T"
                else:
                    Kl_fun_name = Kl_fun
                method_name = type_est + str(K_list)+'/Kl_fun'+Kl_fun_name+'_C'+str(C)
                if type_est == "proposed":
                    method_name += "earlystop"+str(early_stopping)+'/max_iter'+str(max_iter)
            else:
                method_name  = type_est
                
            print('type_est',type_est)
            for seed in seeds:
                os.chdir(path_original+ "/results/")
                print('seed',seed)
                path_name = './trans' + setting +'/N' + str(N) +'/Tnew_' +\
                            str(T_new)+'_type_'+method_name+'_cpitv'+ str(cp_detect_interval)+\
                            '/cov'+str(cov) + '/seed'+str(seed) 
                if os.path.exists(path_name):
                    print('path_name', path_name)
                    os.chdir(path_name)	    	           
                    file_name = "seed_"+str(seed)+".dat"
                    if not os.path.exists(file_name):
                        print('=========== \n BUG! \n============')
                    else:
                        pkl_file = open(file_name, 'rb')
                        t = pickle.load(pkl_file)
                        tmp =[setting, seed, type_est, t['value']['average_reward'],t['value']['discounted_reward']]
                        print('tmp', tmp)
                        res_list.append(tmp)
                        # dis_r.append(t['value']['discounted_reward'])
                        print("t['average_reward']",t['value']['average_reward'])
                        pkl_file.close()

    res_all = np.array(res_list).reshape([-1, 5])
    res_all = pd.DataFrame(res_all)
    print(res_all)
    if is_save:
        os.chdir(path_original+ "/results/")
        if not os.path.exists('value'):
            os.makedirs('value', exist_ok=True)
        file_name="value/vall_"+str(today)+'N' + str(N) + "_1d.csv"          
        res_all.to_csv(file_name, index=False, header=['Setting','seed','init','Average Value', 'Discounted Value'])

#%%
run_collect(is_save)