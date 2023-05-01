##############################################################
#### collect data and save it in csv for R displaying ####
##############################################################
import pickle, os, sys, re
import numpy as np
import pandas as pd
from datetime import date
sys.path.append('/home/huly0209_gmail_com/heterRL')
print(sys.path)
is_save = int(sys.argv[1])
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

#################
def run_collect(is_save):
    os.chdir(path_original+ "/results/")
    print('os.getcwd()',os.getcwd())
    res_list = []
    for setting in trans_setting:
        for type_est in type_all:
            print('type_est',type_est)
            for seed in seeds:
                os.chdir(path_original+ "/results/")
                print('seed',seed)
                path_name = './trans' + setting +'/N' + str(N) +'/Tnew_' +\
                            str(T_new)+'_type_'+type_est+'_cpitv'+ str(cp_detect_interval)+'/cov'+str(cov) + '/seed'+str(seed) 
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