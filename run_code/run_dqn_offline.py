import platform, sys, os, re, pickle
import tensorflow as tf
import numpy as np
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
from joblib import Parallel, delayed
M = 20
method_list = ['oracle']
signal_factor_list = [1, 0.5, 0.1]
train_episodes = 100
test_size_factor = 10
# init_method = 'gmr'
for signal_factor in signal_factor_list:
  for method in method_list:
      for seed in range(M):
          print('seed',seed)
          arg_pass = str(seed) + ' '+ str(signal_factor)+ ' '+ method + ' ' + str(train_episodes) +' ' + str(test_size_factor) 
          runfile('simu/dqn_offline.py',args=arg_pass)
#%%
from joblib import Parallel, delayed
import multiprocessing
num_threads=multiprocessing.cpu_count()

method_list = ['oracle']
M = 20
signal_factor_list = [1]
train_episodes = 100
test_size_factor = 10
def run_one(seed, signal_factor, method):    
    !python simu/dqn_offline.py $seed $signal_factor $method $train_episodes $test_size_factor 
    return
Parallel(n_jobs=num_threads, prefer = 'threads')(delayed(run_one)(seed,signal_factor, method) for seed in range(5,20) for signal_factor in signal_factor_list for method in method_list)

#%% collect results
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
method_list = ['overall','oracle']
res_list= [None] * len(method_list)
for method in method_list:
    value_list =[]
    for seed in range(M):
        os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
        setpath(seed, method)
        file_name = method + '.dat'
        with open(file_name, 'rb') as f:
            t = pickle.load(f)
            value_list.extend([float(t['value'])])
    res_list[method_list.index(method)] = {'method':method,
                                           'value':round(np.mean(value_list),3),
                                           'std':round(np.std(value_list)/np.sqrt(M),3)}
print(res_list)
