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


#%% clustering and change point detection results

nsims = 30
# sim_setting = "split"
type_est = 'proposed'
path_name0 = 'data/'
path_name0 += '20220722/'
effect_sizes = ['strong', 'moderate', 'weak']
settings = [""]
settings_labels = [""]
K_list = range(2,6)
# settings = ["initseparately", "initKmeans", "inittruecluster",
#             "initnoninformative", "initrandom", "inittruecp", "initrandom_selectK"]
# settings_labels = ["Default", "KMeans", "True clustering",
#                    "Changepoints = 1", "Random changepoints", "True changepoints",
#                    "Random and select K from {2,3,4,5,6}"]

df = pd.DataFrame(columns=['Size of Change', 'Seed', 'K_best', 'Change Point Error', 'ARI', '# Iterations']) #'Initialization',
df_best_model = pd.DataFrame(columns=['Size of Change', 'Seed', 'K_best', 'Change Point Error', 'ARI', '# Iterations']) #'Initialization',
row_idx = 0
row_idx_best = 0
for effect_size in effect_sizes:
    print("effect_size:", effect_size)
    path_name = path_name0 + "sim_" + effect_size + "/" + type_est + "/"
    for seed in range(1, nsims + 1):
        IC_best = -1e7
        model_best = None
        init_best = None
        for s in range(len(settings)):
            setting = settings[s]
            # print("setting:", setting)
            file_name = path_name + "seed" + str(seed) + "_" + setting + "_cpresult" + ".dat"
            saved_data = pickle.load(open(file_name, "rb"))
            # if setting != "inittruecluster" and setting != "inittruecp" and saved_data['model'][6][0] > IC_best:
            IC_best = saved_data['model'][6][0]
            K_best = saved_data['K_best']
            # model_best = saved_data
            # init_best = settings_labels[s]
            changepoint_err_best = saved_data["changepoint_err"][K_list.index(K_best)]
            cluster_err_best = saved_data["cluster_err"][K_list.index(K_best)]
            iter_num_best = saved_data["iter_num"]
            df.loc[row_idx] = [effect_size, seed, K_best, saved_data["changepoint_err"], saved_data["cluster_err"],
                               saved_data["iter_num"]] #settings_labels[s],
            row_idx += 1
        df.loc[row_idx] = [effect_size, init_best, seed, changepoint_err_best, cluster_err_best, iter_num_best]
        row_idx_best += 1


    # for s in range(len(settings)):
    #     setting = settings[s]
    #     print("setting:", setting)
    #     # file_path = path_name + type_est + "/"
    #     # read in cp/clustering result
    #     changepoint_err_list = []
    #     cluster_err_list = []
    #     iter_num = []
    #     for seed in range(1,nsims+1):
    #         file_name = path_name + "seed" + str(seed) + "_" + setting + "_cpresult" + ".dat"
    #         saved_data = pickle.load(open(file_name, "rb"))
    #         out[6]
    #         changepoint_err_list.append(saved_data["changepoint_err"])
    #         cluster_err_list.append(saved_data["cluster_err"])
    #         iter_num.append(saved_data["iter_num"])
    #         df.loc[row_idx] = [effect_size, settings_labels[s], seed, saved_data["changepoint_err"], saved_data["cluster_err"], saved_data["iter_num"]+1]
    #         row_idx += 1
    #
    #     print("Changepoint deviation:", round(sum(changepoint_err_list)/len(changepoint_err_list), 3))
    #     print("ARI:", round(sum(cluster_err_list)/len(cluster_err_list), 3))
    #     print("# Iterations:", round(sum(iter_num)/len(iter_num), 3)+1)

print(df)
df.to_csv('output/sim_result_07222021.csv', index = False)
# df_best.to_csv('output/sim_result_best_model_07222021.csv', index = False)

# subprocess.call("Rscript --vanilla p01_plot_cp_cluster.R", shell=True)

# #%% evaluation results
# type_est_list = ['proposed', "oracle", "truecluster", "estimatedcluster", "cusumrl", 'overall']
# type_est_list_label = ['Proposed', "Oracle", "True cluster", "Estimated cluster", "CUSUM-RL", 'Overall']
#
# sim_setting = "split"
# type_est = 'proposed'
# path_name = 'data/'
# path_name += '20220701/'
# path_name += "sim" + sim_setting + "/"
#
# nsims = 30
# K = 3
# N_per_cluster = 25
# values = {}
# for type_est in type_est_list:
#     value_by_type = []
#     if type_est == "cusumrl":
#         for seed in range(1, nsims + 1):
#             mean_by_cluster = [np.random.normal(210.65, 3.91), np.random.normal(205.11, 23.55), np.random.normal(276.68, 2.94)]
#             value_by_type.append(mean_by_cluster)
#     else:
#         file_path = path_name + type_est + "/"
#         for seed in range(1,nsims+1):
#             # print(seed)
#             try:
#                 saved_data = pickle.load(open(file_path + "seed" + str(seed) + "_value.dat", "rb"))
#                 all_values = np.array(saved_data)
#                 mean_by_cluster = []
#                 for k in range(K):
#                     mean_by_cluster.append(np.mean(all_values[k*N_per_cluster:(k+1)*N_per_cluster,:]))
#                 value_by_type.append(mean_by_cluster)
#             except:
#                 continue
#     values[type_est] = np.array(value_by_type)
#
#
# dat = []
# for l in range(len(type_est_list)):
#     data_value = pd.DataFrame(values[type_est_list[l]], columns=['Cluster 1', 'Cluster 2', 'Cluster3'])
#     data_value['Method'] = type_est_list_label[l]
#     dat.append(data_value)
# dat = pd.concat(dat)
# dat.Method = pd.Categorical(dat.Method, categories=type_est_list_label, ordered=True)
# dat.groupby('Method').agg(['mean','std']).round(2)#.mean()
#
# # data_value = pd.DataFrame(values['proposed'], columns = ['Cluster 1', 'Cluster 2', 'Cluster3'])
# # data_value['Method'] = 'Proposed'
# # data_value2 = pd.DataFrame(values['overall'], columns = ['Cluster 1', 'Cluster 2', 'Cluster3'])
# # data_value2['Method'] = 'Overall'
# # data_value3 = pd.DataFrame(np.array([[280.309846, 215.982176, 193.687199]]), columns = ['Cluster 1', 'Cluster 2', 'Cluster3'])
# # data_value3['Method'] = 'CUSUM-RL'
# # data_value = pd.concat([data_value, data_value2, data_value3])
# # data_value.groupby('Method').mean()
#