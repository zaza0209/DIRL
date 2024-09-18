# -*- coding: utf-8 -*-
"""
collect results

@author: test
"""
# collect the results
import sys, os, pickle 
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
path_original = os.getcwd()
#%%
max_iter=int(sys.argv[1])
T=int(sys.argv[2])
N_per_cluster=int(sys.argv[3])
K_true = int(sys.argv[4])
stationary_team_dynamic=int(sys.argv[5])
kappa_interval_step=int(sys.argv[6])
env_type=sys.argv[7]
cov=float(sys.argv[8])
transform_s=int(sys.argv[9])
stationary_changeactionsign=int(sys.argv[10])
effect_size_factor_list =  [float(x) for x in sys.argv[11:]]
#%%
K_list = list(range(1, 5))
threshold_type = 'maxcusum'
M=30
epsilon = 1/T
threshold_type = "maxcusum"
B = 10000
kappa_max=int(0.9*T)
kappa_min =int(0.1*T)
kappa_interval=int(T/kappa_interval_step)
Kl_fun="Nlog(NT)/T"
C=1
early_stopping=0 
ic_T_dynamic=0
p=3
#%%
def setpath():
    if not os.path.exists('results/collect_res'):
        os.makedirs('results/collect_res', exist_ok=True)
    init_name = Kl_fun.replace("/", "_")+"_C"+str(C)+'/max_iter'+str(max_iter)+'_K_list'+str(K_list)
    # if env_type != "original":
    trans_setting_name = str(stationary_team_dynamic)
    if stationary_changeactionsign!=1 and stationary_team_dynamic<0:
        trans_setting_name +='_changesign'
    path_name = 'results/collect_res/stationary_team_dynamic'+ trans_setting_name +'/'+env_type+\
        '/effect_size_factor_' + str(effect_size_factor_list)+\
        '/Ncluster' + str(N_per_cluster) +'_T' + str(T)+'/K'+ str(K_true)+'/cov'+str(cov)+'/init_'+init_name+\
           '/transform_s'+str(transform_s)+ '/kappa_max'+str(kappa_max)+'_min'+str(kappa_min)+'_step'+str(kappa_interval)
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)
    os.chdir(path_name)

#%%
trans_setting_list =["pwconst2", "smooth"]
def get_file_name(init, seed, effect_size_factor, trans_setting):
    if init in ["tuneK_iter","only_clusters"]:
        init_name = init+ Kl_fun.replace("/", "_")+"_C"+str(C)+'/max_iter'+str(max_iter)+'_K_list'+str(K_list)
    else:
        init_name = init
    if transform_s!=0:
        init_name += '/statetrans'
    trans_setting_name = trans_setting + str(stationary_team_dynamic)
    if stationary_changeactionsign!=1 and stationary_team_dynamic<0:
        trans_setting_name+='_changesign'
    if env_type != "original":
        trans_setting_name +='/'+env_type
    path_name = path_original+'/results/trans' + trans_setting_name +'/effect_size_factor_' + str(effect_size_factor)+\
        '/Ncluster' + str(N_per_cluster) +'_T' + str(T)+'/K'+ str(K_true)+'/cov'+str(cov)+'/init_'+init_name+\
            '/kappa_max'+str(kappa_max)+'_min'+str(kappa_min)+'_step'+str(kappa_interval)+\
                 '/seed'+str(seed) +"/seed"+str(seed)+".dat"
    return path_name

init_list = ["tuneK_iter", "cluster_atend", "cp_indi", "only_cp", "only_clusters"]

res_all=[]
def run():
    for trans_setting in trans_setting_list:
        for init in init_list:
            for effect_size_factor in effect_size_factor_list:
                for seed in range(M):
                    file_name = get_file_name(init, seed, effect_size_factor, trans_setting)
                    print('file_name',file_name)
                    if os.path.exists(file_name):
                        print('exist')
                        with open(file_name, "rb") as f:
                            dat = pickle.load(f)
                        tmp = [trans_setting, init, seed, effect_size_factor, dat['cp_err'],dat['ARI']]
            
                        res_all.append(tmp)
                        print(tmp, res_all)
                           
    # Convert the nested dictionary to a DataFrame
    df = pd.DataFrame(res_all, columns= ["setting", "Method", "seed", "Effect size", "Change point error", "ARI"])
    # Display the DataFrame
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)
    print(df.groupby(["setting", "Method","Effect size"]).mean())
    # Calculate the mean
    mean_df = df.groupby(by=["setting", "Method","Effect size"], observed=True).mean()
    
    # Calculate the size of each group
    size_df = df.groupby(by=["setting", "Method","Effect size"], observed=True).size().reset_index(name='size')
    
    # Merge the standard deviation dfFrame with the size dfFrame
    std_df = df.groupby(by=["setting", "Method","Effect size"], observed=True).std().reset_index()
    merged_df = pd.merge(std_df, size_df, on=["setting", "Method","Effect size"])
    
    # Calculate the standard error
    for col in mean_df.columns:
        merged_df[col] = merged_df[col] / np.sqrt(merged_df['size'])
    
    # Drop the 'size' column as it is no longer needed
    se_df = merged_df.drop(columns=['size'])
    
    # Set the index to match mean_df
    se_df.set_index(["setting", "Method","Effect size"], inplace=True)
    
    # Combine the mean and SE into a single dfFrame
    mean_se_df = mean_df.copy()
    
    for col in mean_df.columns:
        mean_se_df[col] = mean_df[col].map('{:.3f}'.format) + ' (' + se_df[col].map('{:.3f}'.format) + ')'
    
    # Print the result
    print(mean_se_df.iloc[:,1:])
    
    
    return df

def box_plot(res_all, save_path=None):
    if len(effect_size_factor_list)==3:
        sorted_effect = sorted(effect_size_factor_list)[-1::-1]
        name_list =["Strong", "Moderate", "Weak"]
        sorted_name = {sorted_effect[i]: name_list[i] for i in range(3)}
        for ind, row in res_all.iterrows():
            res_all.loc[ind, "Effect size"] = sorted_name[row["Effect size"]]
    else:
        sorted_effect = sorted(effect_size_factor_list)[-1::-1]
        name_list = sorted_effect
    
    colors = ['#cc0c00', '#5c88da','#84bd00', '#ffcc00', '#7c878e','#00b5e2','#00af66',"#E69F00","#660099"]

    # Use Seaborn to plot the boxplots
    sns.set(style="whitegrid")  # Set the aesthetic style of the plots
    plt.figure(figsize=(14, 6))  # Set the figure size
    
    # Create a boxplot for 'av_reward'
    plt.subplot(1, 2, 1)  # Add subplot 1 in a 1x2 subplot grid
    ax = sns.boxplot(x='Effect size', y='Change point error', data=res_all, palette=colors,order=name_list)
    ax.set_xlabel('')
    ax.set_ylabel('Change point error')
    ax.grid(False)    
    
    # Create a boxplot for 'dis_reward'
    plt.subplot(1, 2, 2)  # Add subplot 2 in a 1x2 subplot grid
    ax = sns.boxplot(x='Effect size', y='ARI', data=res_all, palette=colors,order=name_list)
    # ax.set_title('Box Plot of Discounted Reward by Policy')
    ax.set_xlabel('')
    ax.set_ylabel('ARI')
    ax.grid(False)    
    
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()  # Display the plots
    
    if save_path:
        # Get the current date and time
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Append the current date to the filename
        filename = f'{save_path}/{current_date}.png'
        print('file_name', filename)
        # Save the figure with the updated filename
        plt.savefig(filename, format='png', dpi=300)
#%%
setpath()
print(os.getcwd())
sys.stdout.flush()
# direct the screen output to a file
stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "w")
print("\nName of Python script:", sys.argv[0])
sys.stdout.flush()
df = run()
save_path = os.getcwd()
print('save_path',save_path)
box_plot(df, save_path=save_path)