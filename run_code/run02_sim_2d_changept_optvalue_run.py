M=20
for seed in range(3,5):
    print('seed',seed)
    runfile('C:/Users/test/Dropbox/tml/IHS/simu/simu/run_code/02_sim_2d_22_changept_optvalue_run.py',
            args=str(seed)+' '+'1')

#%% summerize results
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
overall = []
oracle_cp =[]
oracle_cluster = []
oracle = []
last = []
indi = []
type_est = ['overall', 'oracle_cp','oracle_cluster','oracle','last','indi']

def result_generate(M, file_name):
    value = []
    value_path = np.zeros([int(200/10), M])
    converge = []
    Qerr = []
    for seed in range(M):
        setpath(seed)
        pkl_file = open(file_name, 'rb')
        t = pickle.load(pkl_file)
        pkl_file.close()
        # print(t['path'])
        
        path_tmp = t['path'][:-1]
        value.append(t['final'])
        value_path[:,seed] = np.array(path_tmp)
        converge.append(t['converge'])
        Qerr.append(t['Qerr'])

    value_path_mean = np.nanmean(value_path, axis=1)
    value_mean = np.mean(value)
    tab = {'value_mean':np.mean(value),
           'converge_mean':np.mean(converge),
           'Qerr_mean':np.mean(Qerr)}
    return value_path_mean, tab


for seed in range(5):
    setpath(seed)
    # file_name = "value_overall_gamma_overall" + re.sub("\\.", "", str(gamma)) + ".dat"
    # pkl_file = open(file_name, 'rb')
    # t = pickle.load(pkl_file)
    # # print(t)
    # overall.append(t['final'])
    # pkl_file.close()
    
    # file_name = "value_oracle_gamma_oracle_cp" + re.sub("\\.", "", str(gamma)) + ".dat"
    # pkl_file = open(file_name, 'rb')
    # t = pickle.load(pkl_file)
    # oracle_cp.append(t)
    # pkl_file.close()
    
    file_name = "value_oracle_gamma_oracluster" + re.sub("\\.", "", str(gamma)) + ".dat"
    pkl_file = open(file_name, 'rb')
    t = pickle.load(pkl_file)
    oracle_cluster.append(t['final'])
    pkl_file.close()
    
    # file_name = "value_oracle_gamma_oracle" + re.sub("\\.", "", str(gamma)) + ".dat"
    # pkl_file = open(file_name, 'rb')
    # t = pickle.load(pkl_file)
    # oracle.append(t['final'])
    # pkl_file.close()
    
    # file_name = "value_oracle_gamma_last" + re.sub("\\.", "", str(gamma)) + ".dat"
    # pkl_file = open(file_name, 'rb')
    # t = pickle.load(pkl_file)
    # last.append(t['final'])
    # pkl_file.close()
    
    # file_name = "value_oracle_gamma_indi" + re.sub("\\.", "", str(gamma)) + ".dat"
    # pkl_file = open(file_name, 'rb')
    # t = pickle.load(pkl_file)
    # indi.append(t)
    # pkl_file.close()

import pandas as pd
tab = pd.DataFrame({"type": type_est,
                    "value":[np.mean(overall), np.mean(oracle_cp), np.mean(oracle_cluster), np.mean(oracle), np.mean(last), np.mean(indi)],
                    "std":[np.std(overall)/np.sqrt(M), np.std(oracle_cp)/np.sqrt(M), np.std(oracle_cluster)/np.sqrt(M), np.std(oracle)/np.sqrt(M), np.std(last)/np.sqrt(M), np.std(indi)/np.sqrt(M)]})

tab =round(tab,3)
tab.iloc[:,1] = [str(a)+'('+str(b)+')' for a, b in zip(tab.iloc[:,1], tab.iloc[:,2])]
tab = tab.iloc[:,:2]
# os.chdir("C:/Users/test/Dropbox/tml/IHS/simu/data")
# tab.to_excel('res.xlsx')
print(tab)
