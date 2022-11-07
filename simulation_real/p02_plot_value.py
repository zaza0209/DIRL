'''
Combine p values from multiple simulations using method in
Nicolai Meinshausen, Lukas Meier & Peter Bühlmann (2009) p-Values for
High-Dimensional Regression, Journal of the American Statistical Association, 104:488, 1671-1681,
DOI: 10.1198/jasa.2009.tm08647
First read in saved p-value data from multiple random seeds, and aggregate p-values
with specified quantiles
'''
import platform, sys, os, pickle, re, subprocess
import numpy as np
import pandas as pd
plat = platform.platform()
print(plat)
if plat == 'macOS-12.4-x86_64-i386-64bit' or plat == 'macOS-10.16-x86_64-i386-64bit': ##local
    os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL/simulation_real")
    sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL")
elif plat == 'Linux-3.10.0-1160.42.2.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")
elif plat == 'Linux-3.10.0-1160.6.1.el7.x86_64-x86_64-with-glibc2.17' or plat == 'Linux-3.10.0-1160.45.1.el7.x86_64-x86_64-with-glibc2.17':  # greatlakes
    os.chdir("/home/mengbing/research/HeterRL/simulation_real")
    sys.path.append("/home/mengbing/research/HeterRL")


#%% evaluation results
type_est_list = ['proposed', "oracle", "stathomo", "random"] #"cusumrl",
type_est_list_label = ['Proposed', "Oracle", "Stationary + Homogeneity", "Random"] #"CUSUM-RL",
# type_est_list = ['proposed', "oracle", "truecluster", "estimatedcluster", 'overall'] #"cusumrl",
# type_est_list_label = ['Proposed', "Oracle", "True cluster", "Estimated cluster", 'Overall'] #"CUSUM-RL",

sim_setting = "split"
type_est = 'proposed'
path_name0 = 'data/'
path_name0 += '20220722/'
effect_sizes = ['strong', 'moderate', 'weak']

nsims = 30
K = 3
N_per_cluster = 25
df = pd.DataFrame(columns=['Size of Change', 'Seed', 'Method', 'Cluster', 'Value'])
row_idx = 0
# values = {}
for effect_size in effect_sizes:
    print("effect_size:", effect_size)
    path_name = path_name0 + "sim_" + effect_size + "/"

    for t in range(len(type_est_list)):
        type_est = type_est_list[t]
        type_est_label = type_est_list_label[t]
        value_by_type = []
        # if type_est == "cusumrl":
        #     for seed in range(1, nsims + 1):
        #         mean_by_cluster = [np.random.normal(210.65, 3.91), np.random.normal(205.11, 23.55), np.random.normal(276.68, 2.94)]
        #         value_by_type.append(mean_by_cluster)
        # else:
        file_path = path_name + type_est + "/"
        for seed in range(1,nsims+1):
            # print(file_path + "seed" + str(seed) + "_value.dat")
            try:
                saved_data = pickle.load(open(file_path + "seed" + str(seed) + "_value.dat", "rb"))
                all_values = np.array(saved_data)
                mean_by_cluster = []
                for k in range(K):
                #     mean_by_cluster.append(np.mean(all_values[k*N_per_cluster:(k+1)*N_per_cluster,:]))
                # value_by_type.append(mean_by_cluster)
                    df.loc[row_idx] = [effect_size, seed, type_est_label, k,
                                       np.mean(all_values[k,:])] #[k*N_per_cluster:(k+1)*N_per_cluster,:]
                    row_idx += 1

            except:
                continue
    # values[type_est] = np.array(value_by_type)
print(df)
df.to_csv('output/values_20220722.csv', index = False)
subprocess.call("module load R", shell=True)
subprocess.call("Rscript --vanilla p02_plot_value.R", shell=True)

# dat = []
# for l in range(len(type_est_list)):
#     data_value = pd.DataFrame(values[type_est_list[l]], columns=['Cluster 1', 'Cluster 2', 'Cluster3'])
#     data_value['Method'] = type_est_list_label[l]
#     dat.append(data_value)
# dat = pd.concat(dat)
# dat.Method = pd.Categorical(dat.Method, categories=type_est_list_label, ordered=True)
# dat.groupby('Method').agg(['mean','std']).round(2)#.mean()

# data_value = pd.DataFrame(values['proposed'], columns = ['Cluster 1', 'Cluster 2', 'Cluster3'])
# data_value['Method'] = 'Proposed'
# data_value2 = pd.DataFrame(values['overall'], columns = ['Cluster 1', 'Cluster 2', 'Cluster3'])
# data_value2['Method'] = 'Overall'
# data_value3 = pd.DataFrame(np.array([[280.309846, 215.982176, 193.687199]]), columns = ['Cluster 1', 'Cluster 2', 'Cluster3'])
# data_value3['Method'] = 'CUSUM-RL'
# data_value = pd.concat([data_value, data_value2, data_value3])
# data_value.groupby('Method').mean()
