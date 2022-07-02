
import tensorflow as tf
import numpy as np
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
from joblib import Parallel, delayed
M = 1
method_list = ['overall']
signal_factor_list = [1, 0.5, 0.1]
train_episodes = 1
test_size_factor = 1
for signal_factor in signal_factor_list:
  for method in method_list:
      for seed in range(M):
          arg_pass = str(seed) + ' '+ str(signal_factor)+ ' '+ method + ' ' + str(train_episodes) +' ' + str(test_size_factor)
          runfile('simu/dqn_offline.py',args=arg_pass)
#%%
import platform, sys, os, re, pickle
from joblib import Parallel, delayed
import multiprocessing
num_threads=multiprocessing.cpu_count()-1
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
signal_factor_list = [0.5]
test_size_factor = 10
train_episodes = 2
def run_one(cluster_id, signal_factor):    
    !python simu/dqn_online.py $cluster_id $signal_factor $test_size_factor $train_episodes
    return
Parallel(n_jobs=num_threads, prefer = 'threads')(delayed(run_one)(cluster_id,signal_factor) for cluster_id in range(2) for signal_factor in signal_factor_list)

