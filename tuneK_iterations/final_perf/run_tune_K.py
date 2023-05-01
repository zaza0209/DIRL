# -*- coding: utf-8 -*-
"""
Run tune_K.py 

"""
from joblib import Parallel, delayed
import multiprocessing
import os, pickle
from datetime import datetime
import numpy as np
num_threads=multiprocessing.cpu_count()
#%%
os.chdir("C:/Users/test/Dropbox/DIRL/IHS/simu/simu_original/tuneK_iterations")

M=6
init_list = ['K2']#, 'kmeans']
N = 50
T = 50
trans_setting = 'pwconst2'
K_list =  "1 2 3 4"
effect_size_list=['moderate']
def run_one(seed, init, N, T, trans_setting, effect_size, K_list):    
    !python tune_K.py $seed $init $N $T $trans_setting $effect_size $K_list
    return
# Parallel(n_jobs=num_threads, prefer = 'threads')(delayed(run_one)(seed, init, N, T,
#                                                                   trans_setting,
#                                                                   effect_size,
#                                                                   K_list) 
#                                                  for seed in range(M) 
#                                                  for init in init_list
#                                                  for effect_size in effect_size_list)
effect_size=effect_size_list[0]
# elapsetime=[]
# init = "tuneK_iter"
for seed in range(M):
    print(seed)
    for init in init_list:
        print('init', init)
        starttime = datetime.now()
        run_one(seed, init, N, T, trans_setting, effect_size, K_list)
        # elapsetime.append(datetime.now() - starttime)
        print(datetime.now() - starttime)

#%%
os.chdir("C:/Users/test/Dropbox/DIRL/IHS/simu/simu_original/tuneK_iterations")
K_list =[1, 2, 3, 4]
def setpath(trans_setting, init = "true clustering"):
    os.chdir("C:/Users/test/Dropbox/DIRL/IHS/simu/simu_original/tuneK_iterations")
    path_name = 'results/trans' + trans_setting +'/effect_size_' + effect_size+'/N' + str(N) +'_T' + str(T)+'/K'+ str(K_list)+'/init_'+init+'/seed'+str(seed) 
    os.chdir(path_name)
    
res_all = []
for init in init_list:
    res_all.append({})
    ARI=[]
    cp_err =[]
    runtime=[]
    for seed in range(M):
        setpath(trans_setting, init=init)
        file_name = "seed_"+str(seed)+".dat"
        with open(file_name,'rb') as f:
            t  = pickle.load(f)
            runtime.append(t['elapse'].total_seconds())
            ARI.append(t['ARI'])
            cp_err.append(t['cp_err'])
    res_all[init_list.index(init)]['method'] = init
    res_all[init_list.index(init)]['ARI'] = np.mean(ARI)
    res_all[init_list.index(init)]['ARI_se'] = np.std(ARI)/np.sqrt(M)
    res_all[init_list.index(init)]['cp_err'] = np.mean(cp_err)
    res_all[init_list.index(init)]['cp_err_se'] = np.std(cp_err)/np.sqrt(M)
    res_all[init_list.index(init)]['runtime'] = np.mean(runtime)
    res_all[init_list.index(init)]['runtime_se'] = np.std(runtime)/np.sqrt(M)
# print(np.mean(cp_err))
# print(np.mean(ARI))
# print(np.mean(runtime))
#%% strong effect size
import pandas as pd
file_name = 'C:/Users/test/Dropbox/DIRL/IHS/simu/simu/output/final_perf'
file_name = file_name + '/icmodel_2022-11-02method(7)N50_1d.csv'
old_res = pd.read_csv(file_name)
old_res = old_res[(old_res['init'] == 'best_model') & (old_res['Setting'] == trans_setting)]
old_mean = np.mean(old_res.iloc[:, [3,4]], axis=0)
old_se = np.std(old_res.iloc[:, [3,4]], axis=0)/np.sqrt(M)
# print(old_res)

res_tab = [{'method':'tuneK_iter',
             'cp_err':res_all[0]['cp_err'],
             'cp_err_se':res_all[0]['cp_err_se'],
             'ARI':res_all[0]['ARI'],
             'ARI_se':res_all[0]['ARI_se']},
            {'method':'kmeans',
             'cp_err':old_mean[0],
             'cp_err_se':old_se[0],
             'ARI':old_mean[1],
             'ARI_se':old_se[1]}]

method_list = ['tuneK_iter', 'kmeans']
res_dic=[]
for method in method_list:
    my_dict = res_tab[method_list.index(method)]
    # Create a new dictionary to store combined values
    combined_dict = {}
    # Loop through the keys in the original dictionary
    for key in my_dict.keys():
        # Check if the key ends with '_se'
        if key.endswith('_se'):
            # Extract the corresponding '_eta' key by replacing '_se' with '_eta'
            mean_key = key.replace('_se', '')
            # Check if the '_eta' key exists in the original dictionary
            if mean_key in my_dict:
                # Combine the values and format as a string with 3 digits after the decimal point
                combined_value = '{:.5f}({:.5f})'.format(my_dict[mean_key], my_dict[key])
                
                # Add the combined value to the new dictionary with the key from the original dictionary
                combined_dict[mean_key] = combined_value

    combined_dict['method']=method
    res_dic.append(combined_dict)
    # Convert the combined dictionary to a DataFrame
df = pd.DataFrame(res_dic)
last_column = df.columns[-1]
df = pd.concat([df[last_column], df.drop(last_column, axis=1)], axis=1)


df_pwconst_strong = df
df_pwconst_moderate = df
df_smooth_strong=df

import datetime
# Get the current date and time
now = datetime.datetime.now()
# Format the date and time as a string
date_time_str = now.strftime("%Y-%m-%d")

file_name="C:/Users/test/Dropbox/DIRL/IHS/simu/simu_original/tuneK_iterations/"
file_name = file_name + f"res_{date_time_str}.xlsx"

with pd.ExcelWriter(file_name) as writer:
    df_pwconst_strong.to_excel(writer, sheet_name="pwconst_strong", index=False)
    df_smooth_strong.to_excel(writer, sheet_name="smooth_strong", index=False)
    df_pwconst_moderate.to_excel(writer, sheet_name="pwconst_moderate", index=False)