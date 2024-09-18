##############################################################
#### collect data and save it in csv for R displaying ####
##############################################################
import pickle, os, sys
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime
path_name = os.getcwd()
#%%
max_iter=int(sys.argv[1])
N_per_cluster=int(sys.argv[2])
fixed_kappamin=float(sys.argv[3])
cp_detect_interval=int(sys.argv[4])
true_cp_interval=int(sys.argv[5])
T_new=int(sys.argv[6])
env_type=sys.argv[7]
reward_type=sys.argv[8]
is_transform=int(sys.argv[9])
K_true=int(sys.argv[10])
stationary_team_dynamic=int(sys.argv[11])
stationary_changeactionsign=int(sys.argv[12])
is_r_orignial_scale=int(sys.argv[13])
effect_size_list =[float(i) for i in sys.argv[14:]]
print('is_r_orignial_scale', is_r_orignial_scale)
#%%
M=40
cov=0.25
gamma = 0.9
trans_setting = ['pwconst2','smooth'] 
type_all = ['proposed', 'oracle','only_cp', 'only_clusters', 'overall',
            'cluster_atend', 'cp_indi','cluster_atend2'
            ]
Kl_fun='Nlog(NT)/T' #"sqrtN"
K_list=[1,2,3,4]
early_stopping=1
C=1
T_initial=101
#%%
def setpath():
    if not os.path.exists('results/collect_res'):
        os.makedirs('results/collect_res', exist_ok=True)
    Kl_fun_name = "Nlog(NT)_T"
    method_name = str(K_list) +'/Kl_fun'+Kl_fun_name+'_C'+str(C)
    method_name += "earlystop"+str(early_stopping)+'/max_iter'+str(max_iter)
    if fixed_kappamin < 1:
        method_name += '/kappamin'+str(fixed_kappamin)
    
    if is_transform != 1:
        method_name +='/transform'+str(is_transform)
    trans_setting_name ="/effect_size"+str(effect_size_list)
    if env_type != "randomstationary":
        trans_setting_name += '/'+env_type
    if reward_type != "next_state":
        trans_setting_name += '/reward'+reward_type
    if K_true!=3:
        trans_setting_name+='K_true'+str(K_true)
    if stationary_team_dynamic!=4 and K_true>2:
        trans_setting_name+= '/stationary'+str(stationary_team_dynamic)
        if stationary_team_dynamic<1 and stationary_changeactionsign!=1:
            trans_setting_name+= "changesign"

    data_path = 'results/collect_res/'+trans_setting_name+'N' + str(N_per_cluster) +'/Tnew_' +\
                str(T_new)+'_type_'+method_name+'_cpitv'+ str(cp_detect_interval)+'/true_cp'+str(true_cp_interval)+\
                '/cov'+str(cov)  
    if is_r_orignial_scale:
        data_path += "/originalR"+str(is_r_orignial_scale)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)
    
#%%
def transform_back(x):
    return (x*0.1*4.141009340555169+19.343480280896884)**3

def get_first_2clusters_value(raw_reward):
    if not is_r_orignial_scale:
        transformed_reward = transform_back(raw_reward)
    else:
        transformed_reward = raw_reward.copy()
    av_r = np.mean(transformed_reward[:-1*N_per_cluster, T_initial:])
    dis_r = 0.0
    for t in range(T_initial, transformed_reward.shape[1]):
        dis_r += transformed_reward[:-1*N_per_cluster,t] * gamma**(t - T_initial)
    # print('T_initial', T_initial, av_r, np.mean(dis_r), transformed_reward.shape[1])
    return av_r, np.mean(dis_r)
    
def get_value_in_originalR(raw_reward):
    if not is_r_orignial_scale:
        transformed_reward = transform_back(raw_reward)
    else:
        # print('not transformed')
        transformed_reward = raw_reward.copy()
    av_r = np.mean(transformed_reward[:, T_initial:])
    dis_r = 0.0
    for t in range(T_initial, transformed_reward.shape[1]):
        dis_r += transformed_reward[:,t] * gamma**(t - T_initial)
    # print('T_initial', T_initial, av_r, np.mean(dis_r), transformed_reward.shape[1])
    return av_r, np.mean(dis_r)
    
def run_collect():
    res_all = []
    sorted_effect = sorted(effect_size_list)[-1::-1]
    if len(effect_size_list)==3:
        effect_name = ["Strong", "Moderate", "Weak"]
        effect_name_dict = {sorted_effect[i]:effect_name[i] for i in range(3)}
    else:
        effect_name = sorted_effect
        effect_name_dict= {i:i for i in sorted_effect}
    type_name = {'proposed': 'DIRL', 'oracle':'Oracle','overall': "DHRL",
                   "only_cp":'Homogeneous', 'only_clusters':'Stationary',
                    'cluster_atend': 'TCD', 'cp_indi':'ITCPD',
                    'cluster_atend2':'TCD2'
                   }
    for setting in trans_setting:
        for effect_size_factor in effect_size_list:
            for type_est in type_all:
                if type_est in ["proposed", "only_clusters", "cluster_atend",'cluster_atend2']:
                    if Kl_fun=='Nlog(NT)/T':
                        Kl_fun_name = "Nlog(NT)_T"
                    else:
                        Kl_fun_name = Kl_fun
                    method_name = type_est + str(K_list)+'/Kl_fun'+Kl_fun_name+'_C'+str(C)
                    if type_est == "proposed":
                        method_name += "earlystop"+str(early_stopping)+'/max_iter'+str(max_iter)
                else:
                    method_name  = type_est
                if type_est in ["proposed", "only_cp"]:
                    if fixed_kappamin < 1:
                        method_name += '/kappamin'+str(fixed_kappamin)
                if is_transform != 1 and type_est in ["proposed", "only_clusters", "only_cp", "cluster_atend"]:
                    method_name +='/transform'+str(is_transform)
                if effect_size_factor != 1.0:
                    trans_setting_name =setting + "/effect_size"+str(effect_size_factor)
                else:
                    trans_setting_name = setting
                if env_type != "randomstationary":
                    trans_setting_name += '/'+env_type
                if reward_type != "next_state":
                    trans_setting_name += '/reward'+reward_type
                if K_true!=3:
                    trans_setting_name+='K_true'+str(K_true)
                if stationary_team_dynamic!=4 and K_true>2:
                    trans_setting_name+= '/stationary'+str(stationary_team_dynamic)
                    if stationary_team_dynamic<1 and stationary_changeactionsign!=1:
                        trans_setting_name+= "changesign"
    
                print('type_est',type_est)
                for seed in range(M):
                    file_name = path_name+ '/results/trans' + trans_setting_name +'/N' + str(N_per_cluster) +'/Tnew_' +\
                                str(T_new)+'_type_'+method_name+'_cpitv'+ str(cp_detect_interval)+'/true_cp'+str(true_cp_interval)+\
                                '/cov'+str(cov) + '/seed'+str(seed)  
                    if is_r_orignial_scale:
                        file_name += '/seed_'+str(seed)+'_originalR.dat'
                    else:
                        file_name += '/seed_'+str(seed)+'.dat'
                    print(file_name)
                    if os.path.exists(file_name):
                        print('file_name', file_name)
                        with open(file_name, 'rb') as f:
                            t = pickle.load(f)
                        raw_reward =t['value']['raw_reward']
                        av_r_original, dis_r_original = get_value_in_originalR(raw_reward)
                        av_r_first2, dis_r_first2  =get_first_2clusters_value(raw_reward)
                        tmp =[setting, effect_name_dict[effect_size_factor], seed, type_name[type_est], t['value']['average_reward'],t['value']['discounted_reward'],
                              av_r_original, dis_r_original, av_r_first2, dis_r_first2]
                        print('tmp', tmp)
                        res_all.append(tmp)
                        sys.stdout.flush()
    if len(res_all)>0:
        res_all = pd.DataFrame(res_all)
        res_all.columns = ['Setting','Effect size', 'seed', 'Policy', 'Average reward', 'Discounted reward', 
                           'Average reward (orignal scale)', 'Discounted reward (original scale)',
                           'Average reward (first 2)', 'Discounted reward (first 2)']
        res_all['Effect size'] = pd.Categorical(res_all['Effect size'], categories=effect_name, ordered=True)
        res_all['Policy'] = pd.Categorical(res_all['Policy'], categories=type_name.values(), ordered=True)
        
        # pd.set_option('display.max_rows', None)
        # pd.set_option('display.max_columns', None)
        # print(res_all)
        
    return res_all

def box_plot(data, plot_type='Average reward', save_path=None):
    ## plot value
    if plot_type in ["Average reward", 'Discounted reward', "Average reward (original scale)", 'Discounted reward (original scale)']:
        selected_data = data[data['av_dis']==plot_type]
        y_var = 'Value'
        custom_labels = list(data['Policy'].cat.categories)
    elif plot_type in ['Average reward difference', 'Discounted reward difference', "Average reward (original scale) difference", 'Discounted reward (original scale) difference']:
        selected_data = data[(data['av_dis']==plot_type[:-11]) & (data['Policy'] != "DIRL")]
        cat_list = list(selected_data["Policy"].unique()) #  list(filtered_df["Policy"].cat.categories)
        # order = [cat for cat in cat_list if selected_data['Policy'].str.contains(cat).any()]
        selected_data['Policy'] = pd.Categorical(selected_data['Policy'], categories=cat_list, ordered=True)
        y_var = 'Value Difference'
        custom_labels = ['DIRL-Oracle', 'DIRL-DHRL', 'DIRL-Homogeneous', 'DIRL-Stationary',
                          'DIRL-TCD', 'DIRL-TCPD','DIRL-TCD2'
                         ]
    else:
        raise ValueError('Invalid plot_type')
        
    colors = ['#cc0c00', '#5c88da','#84bd00', '#ffcc00', '#7c878e','#00b5e2','#00af66',"#E69F00","#660099"]

    g =sns.catplot(
    data=selected_data , x='Policy', y=y_var,
    col='Effect size', kind='box', col_wrap=3,
    hue="Policy", palette=colors,
    )
    for ax in g.axes.flat:
        ax.set_xticklabels([])
        ax.grid(True, axis='y')
        if plot_type[-11:] == 'difference':
            ax.axhline(color="red", ls="--")
    g.set_axis_labels("", plot_type)
    
        
    
    handles, labels = g.axes[0].get_legend_handles_labels()
    labels = custom_labels
    g.add_legend(title="Policy", labels=labels, handles=handles)
    plt.show()
    
    if save_path:
        # Get the current date and time
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Append the current date to the filename
        filename = f'{save_path}/{current_date}_{plot_type}.png'
        # Save the figure with the updated filename
        plt.savefig(filename, format='png', dpi=300)
        
        
#%%
if __name__ == "__main__":
    setpath()
    print(os.getcwd())
    sys.stdout.flush()
    # direct the screen output to a file
    stdoutOrigin = sys.stdout
    sys.stdout = open("log.txt", "w")
    print("\nName of Python script:", sys.argv[0])
    sys.stdout.flush()
    data = run_collect()
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)  
    pd.set_option('display.width', 1000) 
    print(data)
    #%% print value
    # Calculate the mean
    mean_df = data.groupby(by=['Setting','Effect size', 'Policy'], observed=True).mean()
    
    # Calculate the size of each group
    size_df = data.groupby(by=['Setting','Effect size', 'Policy'], observed=True).size().reset_index(name='size')
    
    # Merge the standard deviation DataFrame with the size DataFrame
    std_df = data.groupby(by=['Setting', 'Effect size','Policy'], observed=True).std().reset_index()
    merged_df = pd.merge(std_df, size_df, on=['Setting', 'Effect size','Policy'])
    
    # Calculate the standard error
    for col in mean_df.columns:
        merged_df[col] = merged_df[col] / np.sqrt(merged_df['size'])
    
    # Drop the 'size' column as it is no longer needed
    se_df = merged_df.drop(columns=['size'])
    
    # Set the index to match mean_df
    se_df.set_index(['Setting', 'Effect size','Policy'], inplace=True)
    
    # Combine the mean and SE into a single DataFrame
    mean_se_df = mean_df.copy()
    
    for col in mean_df.columns:
        mean_se_df[col] = mean_df[col].map('{:.3f}'.format) + ' (' + se_df[col].map('{:.3f}'.format) + ')'
    
    # Print the result
    # print(mean_se_df.iloc[:,1:])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)  
    pd.set_option('display.width', 1000) 
    if is_r_orignial_scale:
        print(mean_se_df)
        # print(mean_se_df.loc[:,['Average reward', 'Discounted reward', 
        #             'Average reward (first 2)', 'Discounted reward (first 2)']])
    else:
        print(mean_se_df)
    sys.stdout.flush()
    
    #%% calculate value diff
        # data["Average reward (original scale)"] = transform_back(data["Average reward"])
    with open('data.dat', 'wb') as f:
        pickle.dump(data, f)
    data = pd.melt(data, id_vars=['Setting', "Policy", "seed", 'Effect size'], 
                    value_vars=['Average reward', 'Discounted reward', 
                                'Average reward (orignal scale)', 'Discounted reward (original scale)',
                                'Average reward (first 2)', 'Discounted reward (first 2)'],
                    var_name='av_dis', value_name='Value')
    if not data.index.is_unique:
        print("The index is not unique. Resetting the index.")
        data = data.reset_index(drop=True)
    reference_policy = "DIRL"
    reference_df = data[data['Policy'] == reference_policy]
    # Calculate differences
    for index, row in data.iterrows():
        if row['Policy'] != reference_policy:
            # Get the reference row with the same seed, Setting, and av_dis
            ref_row = reference_df[(reference_df['seed'] == row['seed']) & 
                                   (reference_df['Setting'] == row['Setting']) & 
                                   (reference_df['av_dis'] == row['av_dis']) &
                                   (reference_df['Effect size'] == row['Effect size'])]
            if not ref_row.empty:
                ref_row = ref_row.iloc[0]
                av_dis_diff = ref_row['Value']-row['Value'] 
                # Add the calculated differences to the new columns
                data.at[index, 'Value Difference'] = av_dis_diff
            else:
                data.at[index, 'Value Difference'] =np.nan
        else:
            data.at[index, 'Value Difference'] =0
    # Convert the difference columns to numeric types
    data['Value Difference'] = pd.to_numeric(data['Value Difference'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)  
    pd.set_option('display.width', 1000)          

    if is_r_orignial_scale:
        print(data.iloc[:,:2])
    else:
        print(data)
    with open('plot_data.dat', 'wb') as f:
        pickle.dump(data, f)
    
    #%% print value diff
    # Group by Setting and Policy, and then calculate mean and standard error for each av_dis category
    results = data.groupby(['Setting','Effect size', 'Policy', 'av_dis'], observed=True).agg(
        avg_diff=('Value Difference', 'mean'),
        se_diff=('Value Difference', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))
    ).reset_index()
    
    # Pivot the table to display av_dis categories as columns
    pivot_tables = {}
    settings = results['Setting'].unique()
    Effects = results['Effect size'].unique()
    for setting in settings:
        for effect in Effects:
            setting_df = results[(results['Setting'] == setting) & (results['Effect size'] == effect)]
            pivot_table = setting_df.pivot(index='Policy', columns='av_dis', values=['avg_diff', 'se_diff'])
            
            # Combine avg_diff and se_diff into a single string
            combined = pivot_table.apply(lambda x: x['avg_diff'].map('{:.4f}'.format) + ' (' + x['se_diff'].map('{:.4f}'.format) + ')', axis=1)
            
            # Store the formatted table
            pivot_tables[(setting, effect)] = combined
    
    # Display the tables
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)  
    pd.set_option('display.width', 1000) 
    for setting, table in pivot_tables.items():
        print(f"\nSetting: {setting}")
        if is_r_orignial_scale:
            print(table.loc[:,['Average reward', 'Discounted reward', 
                        'Average reward (first 2)', 'Discounted reward (first 2)']])
        else:
            print(table)
    
    #%% generate plot
    box_plot(data, "Average reward", os.getcwd())
    box_plot(data, "Discounted reward", os.getcwd())
    box_plot(data, 'Average reward difference', os.getcwd())
    box_plot(data, 'Discounted reward difference', os.getcwd())
    box_plot(data, "Average reward (original scale) difference", os.getcwd())
    box_plot(data, 'Discounted reward (original scale) difference', os.getcwd())
     
