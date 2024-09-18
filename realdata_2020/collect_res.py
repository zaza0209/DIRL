# -*- coding: utf-8 -*-
"""
collect results

@author: test
"""
# collect the results
import platform, sys, os, pickle, re
script_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels to reach the "ClusterRL" directory
dirl_dir = os.path.abspath(os.path.join(script_dir, '..'))

# Append the ClusterRL directory to sys.path
sys.path.append(dirl_dir)
import numpy as np
import pandas as pd
from datetime import date
path_name = os.getcwd()
#%%
seed = int(sys.argv[1])
state_transform = int(sys.argv[2])
early_stopping = int(sys.argv[3])
max_iter=int(sys.argv[4])
threshold_type=sys.argv[5]
ic_T_dynamic=int(sys.argv[6])
test_cluster_type = sys.argv[7]
is_cv =int(sys.argv[8])
C=int(sys.argv[9]) if sys.argv[9]=="1" else float(sys.argv[9]) 
T_train_len=int(sys.argv[10])
is_r_orignial_scale=int(sys.argv[11])
T_train_list = [int(sys.argv[i]) for i in range(12, 12+T_train_len)]
rbf = False

B = 10000

Kl_fun="Nlog(NT)/T"

    
K_list = list(range(1,5))
gamma=0.9
p_var = 3
#%%
def setpath():
    Kl_fun_name = "Nlog(NT)_T"
    method_name = str(K_list)+'/Kl_fun'+Kl_fun_name+'_C'+str(C)
    method_name += "earlystop"+str(early_stopping)+'/max_iter'+str(max_iter)
   
    path_name = 'results/collect_res/T_train_list'+str(T_train_list)+'/threshold'+threshold_type+\
        "/"+method_name+'/is_cv'+str(is_cv)+"/seed"+str(seed)

    
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    os.chdir(path_name)
    print(os.getcwd())
    sys.stdout.flush()
    return path_name

#%%

def run():
    method_list =["proposed", "only_cp", "only_clusters","overall"]
    method_name_list = {"proposed":"proposed",
                        "only_cp":"Homongeneous",
                        "only_clusters":"Stationary",
                        "overall":"DH"
                        }
    method_res = {
        name:{
            T_train:None
            for T_train in T_train_list
            } 
                  for name in method_name_list.values()
                  }
    for method in method_list:
        for T_train in T_train_list:
            kappa_min = int(0.1*T_train)
            kappa_max = T_train
            if method in ["proposed", "only_clusters"]:
                Kl_fun_name = "Nlog(NT)_T"
                method_name = method + str(K_list)+'/Kl_fun'+Kl_fun_name+'_C'+str(C)
                if ic_T_dynamic:
                    method_name += "_Tdynamic"
                if method == "proposed":
                    method_name += "earlystop"+str(early_stopping)+'/max_iter'+str(max_iter)
                sys.stdout.flush()
            else:
                method_name  = method
           
            file_name = path_name+'/results'+'/T_train'+str(T_train)+'/threshold'+threshold_type+\
                '/'+method_name+'/kappa_min'+str(kappa_min)+'_max'+str(kappa_max)+"/seed"+str(seed)
            if test_cluster_type == "proposed_g_index" and method != "proposed":
                test_cluster_type_name = test_cluster_type + "earlystop"+str(early_stopping)+'/max_iter'+str(max_iter)
            else:
                test_cluster_type_name = test_cluster_type + "earlystop"+str(early_stopping)
                
            file_name += '/'+test_cluster_type_name+'/is_cv'+str(is_cv)+'_Tdynamic'+str(ic_T_dynamic)+"/"
            if is_r_orignial_scale:
                file_name += "seed_origianlR.dat"
            else:
                file_name += "seed.dat"
            print(file_name)
            if os.path.exists(file_name):
                with open(file_name, "rb") as f:
                    dat = pickle.load(f)
                method_res[method_name_list[method]][T_train] = dat["ope_value"]
        
        method_res[method_name_list[method]]["Average"] = np.mean([method_res[method_name_list[method]][i] 
                                                 for i in T_train_list if method_res[method_name_list[method]][i] != None])
                           
    # Convert the nested dictionary to a DataFrame
    df = pd.DataFrame.from_dict(method_res, orient='index')
    
    # Display the DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)
    print(df["Average"])
    sys.stdout.flush()

    return
#%%
setpath()
# direct the screen output to a file
stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "w")
print("\nName of Python script:", sys.argv[0])
run()