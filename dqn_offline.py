import platform, sys, os, re, pickle
import tensorflow as tf
import numpy as np
plat = platform.platform()
# print(plat)
if plat == 'Windows-10-10.0.14393-SP0': ##local
    os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
    sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
elif plat == 'Linux-5.4.188+-x86_64-with-Ubuntu-18.04-bionic': # colab
    os.chdir('/content/drive/My Drive/Colab Notebooks/heterRL')
    sys.path.append('/content/drive/My Drive/Colab Notebooks/heterRL')
elif plat == 'Linux-3.10.0-1160.6.1.el7.x86_64-x86_64-with-glibc2.17' or plat == 'Linux-3.10.0-1160.45.1.el7.x86_64-x86_64-with-glibc2.17':  # greatlakes
    os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_real")
    sys.path.append("/home/mengbing/research/RL_nonstationary/code2/")
elif plat == 'Linux-3.10.0-1160.15.2.el7.x86_64-x86_64-with-centos-7.9.2009-Core': #gcp
    os.chdir("/home/limengbinggz_gmail_com/rl_nonstationary/simulation_nonstationary_real")
    sys.path.append("/home/limengbinggz_gmail_com/rl_nonstationary/")
#%%
from tensorflow import keras
from collections import deque
# import time
import random
from scipy.stats import multivariate_normal
# os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
# sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
import simu.simulate_data_pd as sim
# %% 
seed = int(sys.argv[1])
signal_factor= float(sys.argv[2])
method = sys.argv[3]
# An episode a full game
train_episodes = int(sys.argv[4])
test_size_factor = int(sys.argv[5])
tf.random.set_seed(seed)
np.random.seed(seed)
#parameters to simulate data
N=36
# terminal timestamp
T = 100
# dimension of X0
p = 2
# mean vector of X0
mean0 = 0
# diagonal covariance of X0
cov0 = 0.5
# mean vector of random errors zt
mean = 0
# diagonal covariance of random errors zt
cov = 0.25
trans_setting = 'pwconst2'
reward_setting = 'homo'

gamma = 0.9
# MIN_REPLAY_SIZE = 300
coef1=[[signal_factor*0, signal_factor*0, signal_factor*-0.25],[signal_factor*0, signal_factor*0, signal_factor*0.25]]
coef =[coef1,coef1[::-1]]# this is acceptable 0609
signal = [[signal_factor*0, signal_factor*0], [signal_factor*0, signal_factor*0]]
nrep = N*test_size_factor

#%% environment setup
def setpath(seed):
    # os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
    append_name = '_N' + str(N) + '_1d'
    if not os.path.exists('results/value_evaluation'):
        os.makedirs('results/value_evaluation', exist_ok=True)
    data_path = 'results/value_evaluation/coef'+re.sub("\\ ", "",re.sub("\\.", "", re.sub("\\]","", re.sub("\\[", "", re.sub("\\, ", "", str(coef1))))))+'/sim_result_trans' + trans_setting + '_reward' + reward_setting + '_gamma' + re.sub("\\.", "", str(gamma)) + \
                                                 append_name
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    data_path += '/sim_result' + method + '_gamma' + re.sub("\\.", "", str(gamma)) + \
                 append_name + '_' + str(seed)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)
    return
# setpath(seed)

def save_data(file_name, estimated_value,std_value,converge,model=None, model2=None):
    setpath(seed)
    # print('save data')
    if model is not None and model2 is None:
        with open(file_name, "wb")as f:
            pickle.dump({'value':estimated_value,
                          'std':std_value,
                         # 'path':estimated_value_path,
                          'converge':converge,
                         'weights':[model.get_weights()]},f)
    elif model is not None and model2 is not None:
        with open(file_name, "wb")as f:
            pickle.dump({'value':estimated_value,
                          'std':std_value,
                         # 'path':estimated_value_path,
                          'converge':converge,
                         'weights':[model.get_weights(), model2.get_weights()]},f)
    else:
        with open(file_name, "wb")as f:
            pickle.dump({'value':estimated_value,
                          'std':std_value,
                         # 'path':estimated_value_path,
                          'converge':converge,
                         'weights':None},f)
 
    # model.save('model')
    return

# direct the screen output to a file
# stdoutOrigin = sys.stdout
# sys.stdout = open("results/value_evaluation/log_" + method + ".txt", "w")
# print("\nName of Python script:", sys.argv[0])
# sys.stdout.flush()
#%% environment
def generate_initial_state():
    St = np.ones(2)
    St[0] = np.random.normal(mean0, cov0)  # step count of week t
    St[1] = np.random.normal(mean0, cov0)  # sleep of week t
    # St[2] = np.random.normal(7, 1)  # mood score of week t
    return St

def transition(St, At, transition_matrix):
    '''
    :param k: index of cluster k
    :param t: time point t
    :return:
    '''
    # St_full = np.insert(St, 0, 1, axis=0)
    err_mean = np.zeros(2)
    err_cov = np.diag([cov,cov])
    return transition_matrix[At] @ St + \
           multivariate_normal.rvs(err_mean, err_cov)

def get_reward(St):
    return St[0]

def policy(state, model):
    return np.argmax(get_qs(model, state))

def reward(model, episodes=500, T=100):
    rewards = 0
    for i in range(episodes):
        print(i)
        state = generate_initial_state()
        for t in range(T):
            a = policy(state, model)
            r = get_reward(state)
            state = transition(state, a, transition_matrix)
            rewards += r 
    rewards = rewards / episodes
    return rewards
#%% agent
def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    Q function
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

def get_qs(model, state):
    return model.predict(state.reshape([1, state.shape[0]]), verbose=0)[0]

def train(discount_factor, replay_memory, model, target_model,MIN_REPLAY_SIZE=300):
    learning_rate = 0.7  # Learning rate
    # MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    # print('train')
    batch_size = int(MIN_REPLAY_SIZE/8)
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states, verbose=0)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states, verbose=0)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


#%% main
def run(States, Actions, Rewards, MIN_REPLAY_SIZE=300, tol = 1e-5):
    if type(States) == list:
        N = len(States)
    else:
        N = States.shape[0]

    # 1. Initialize the Target and Main models
    state_space_shape = (2,)
    # Main Model (updated every 4 steps)
    model = agent(state_space_shape, 2)
    # Target Model (updated every 100 steps)
    target_model = agent(state_space_shape, 2)
    target_model.set_weights(model.get_weights())
    target_old = agent(state_space_shape, 2)
    target_old.set_weights(model.get_weights())
    
    steps_to_update_target_model = 0
    converge=0
    
    for episode in range(train_episodes):
        print('episode', episode)
        replay_memory = deque(maxlen=50_000)
        steps_to_update_target_model=0
        for i in range(N):
            if type(States) == list:
                states = States[i]
                rewards = Rewards[i]
                actions = Actions[i]
            else:
                states = States[i,:,:]
                rewards = Rewards[i, :]
                actions = Actions[i, :]
            # print('i',i)
            T = rewards.shape[0]
            for t in range(T):
                # print('t',t)
                steps_to_update_target_model += 1
                sars1 = []
                sars1.append(states[t,:])
                sars1.append(actions[t])
                sars1.append(rewards[t])
                sars1.append(states[t+1,:])
                sars1.append(int(t==T-1))
                replay_memory.append(sars1)
                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 4 == 0:
                    train(gamma, replay_memory, model, target_model, MIN_REPLAY_SIZE)
                
                
                if steps_to_update_target_model >= 99 and len(replay_memory) > MIN_REPLAY_SIZE :
                    print('Copying main network weights to the target network weights')
                    target_old.set_weights(target_model.get_weights())
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                    # err = 0.0
                    # no need to break here, offline data
                    # for i in range(len(model.layers)):
                    #     err += np.linalg.norm(target_model.get_weights()[i]-target_old.get_weights()[i], ord=2)
                    # print('err',err)
                    # if err < tol:
                    #     converge=1
                    #     break
            if converge:
                break
        if converge:
            break
        # print('Total training rewards: {} after n steps = {} with final reward = {}'.format(
        #     cum_rewards, episode, reward))
    if train_episodes == 0:
        episode = -1
    return target_model,episode


#%% gen data
def gen_dat(N, T, coef, signal, changepoint_list=None, seed=1):
    # np.random.seed(seed)
    if changepoint_list is None:
        changepoint_list = [int(T/2) + int(0.1 * T) - 1,int(T/2) - int(0.1 * T) - 1] 
    w = 0.01
    delta = 1 / 10
    changepoints_true = np.zeros([N, 1])
    States = np.zeros([N, T, p])
    Rewards = np.zeros([N, T-1])
    Actions = np.zeros([N, T-1])
    coef_tmp = [None] * 2
    changepoint = 0
    for i in range(N):
        if i < int(N/2):
            changepoint = changepoint_list[0]
            coef_tmp[0] = coef[0][0]
            coef_tmp[1] = coef[1][0]
            signal_tmp = [signal[0][0], signal[1][0]]

        else:
            changepoint = changepoint_list[1]
            coef_tmp[0] = coef[0][1]
            coef_tmp[1] = coef[1][1]
            signal_tmp = [signal[0][1], signal[1][1]]
            
        sim_dat = sim.simulate_data(1, T, p, changepoint, delta)
        # print(trans_setting, reward_setting)
        if trans_setting == 'homo' and reward_setting == 'pwconst2':
            def mytransition_function(t):
                return sim_dat.transition_homo(mean, cov)
            def myreward_function(t):
                return sim_dat.reward_pwconstant2(t)
        elif trans_setting == 'homo' and reward_setting == 'smooth':
            def mytransition_function(t):
                return sim_dat.transition_homo(mean, cov)
            def myreward_function(t):
                return sim_dat.reward_smooth2(t, w)
        elif trans_setting == 'pwconst2' and reward_setting == 'homo':
            def mytransition_function(t):
                return sim_dat.transition_pwconstant2(t, mean, cov, coef_tmp, signal_tmp)
            def myreward_function(t):
                return sim_dat.reward_homo()
        elif trans_setting == 'smooth' and reward_setting == 'homo':
            def mytransition_function(t):
                return sim_dat.transition_smooth2(t, mean, cov, w)
            def myreward_function(t):
                return sim_dat.reward_homo()
        States0, Rewards0, Actions0 = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function)
        States[i, :, :] = States0
        Rewards[i, :] = Rewards0
        Actions[i, :] = Actions0
        changepoints_true[i, ] = changepoint
    # normalize state variables
    # def transform(x):
    #     return (x - np.mean(x)) / np.std(x)
    # for i in range(p):
    #     States[:,:,i] = transform(States[:,:,i])
    g_index_true = np.hstack([np.zeros(int(N/2)), np.ones(int(N/2))])
    Actions = Actions.astype(int)
    g_index_true = g_index_true.astype(int)
    return States, Rewards, Actions, changepoints_true, g_index_true

States, Rewards, Actions, changepoints_true, g_index_true = gen_dat(N, T, 
                                                      coef, signal,None,None)
#%% value 
def simulate(transition_matrix, N=25, optimal_policy_model = None, S0=None, A0=None, 
             T0=0, T1=T):
    '''
    simulate data after change points
    '''
    w = 0.01
    delta = 1 / 10

    sim_dat = sim.simulate_data(N, T, p, T0, delta)
    if trans_setting == 'homo' and reward_setting == 'pwconst2':
        def mytransition_function(t):
            return sim_dat.transition_homo(mean, cov)
        def myreward_function(t):
            return sim_dat.reward_pwconstant2(t)
    elif trans_setting == 'homo' and reward_setting == 'smooth':
        def mytransition_function(t):
            return sim_dat.transition_homo(mean, cov)
        def myreward_function(t):
            return sim_dat.reward_smooth2(t, w)
    elif trans_setting == 'pwconst2' and reward_setting == 'homo':
        def mytransition_function(t):
            return sim_dat.transition_pwconstant2(t, mean, cov, coef=None, signal=None, transition_matrix = transition_matrix)
        def myreward_function(t):
            return sim_dat.reward_homo()
    elif trans_setting == 'smooth' and reward_setting == 'homo':
        def mytransition_function(t):
            return sim_dat.transition_smooth2(t, mean, cov, w)
        def myreward_function(t):
            return sim_dat.reward_homo()
    # States, Rewards, Actions = sim_dat.simulate(mean0, cov0, mytransition_function, myreward_function,
    #                                              T0 = T0, T1 = T1, optimal_policy_model=optimal_policy_model)
    States = np.zeros([N, T1-T0 + 1, p])  # S[i][t]
    Rewards = np.zeros([N, T1-T0])  # R[i,t]
    Actions = np.zeros([N, T1-T0])  # Actions[i,t]

    if not S0 is None:
        States[:, 0, :] = S0

    if not A0 is None:
        Actions[:, 0] = A0

    # myState = np.zeros([1, 2, p])

    t = 0  # time 0
    for i in range(N):  # for each individual

        # generate initial state S_i0
        if S0 is None:
            St = generate_initial_state()
            # print(self.St)
            States[i, 0, :] = St
        else:
            St = States[i, 0, :]

        # generate action
        # myState[0, 0, :] = St
        # print('myState', myState, 'shape', myState.shape)
        At = np.argmax(optimal_policy_model.predict(St.reshape([1, St.shape[0]]), verbose=0))
        Actions[i, t] = At

        # generate immediate response R_i,t
        sim_dat.St = St
        sim_dat.At = At
        Rewards[i, t] = myreward_function(t + T0)
        St = mytransition_function(t + T0)
        States[i, t + 1, :] = St

    # other times
    for i in range(N):  # each episode i
        # print(i)i=0
        St = States[i, 1, :]
        for t in range(1, T1-T0):  # each time point t
            # print('t',t)
            # generate policy
            At = np.argmax(optimal_policy_model.predict(St.reshape([1, St.shape[0]]), verbose=0))
            Actions[i, t] = At

            # generate immediate response R_i,t
            sim_dat.St = St
            sim_dat.At = At
            Rewards[i, t] = myreward_function(t+T0)
            # print(Rewards[i, t])
            
            # generate S_i,t+1
            St = mytransition_function(t+T0)
            States[i, t+1, :] = St

    # convert Actions to integers
    Actions = Actions.astype(int)

    return States, Rewards, Actions
def calculate_value(transition_matrix, optimal_policy_model, transition_matrix, nrep, T1_interval):
    _, Rewards_new, _, = simulate(transition_matrix, 
                                  nrep, optimal_policy_model=optimal_policy_model, T0=0, T1 = int(T1_interval))
    est_v = 0.0
    for t in range(T1_interval):
        est_v += Rewards_new[:,t] * gamma**t
    return est_v
#%% overall
if method == "overall":
    target_model_overall, episode = run(States, Actions, Rewards)
    opt_values_overall = []
    for i in range(nrep):
        observation = generate_initial_state()
        encoded_reshaped = observation.reshape([1, observation.shape[0]])
        predicted = target_model_overall.predict(encoded_reshaped, verbose=0).flatten()
        opt_values_overall.append(max(predicted))
    opt_values_overall = calculate_value(target_model_overall)
    print("overall value:", np.mean(opt_values_overall)) #signal0.5: 0.17776427
    file_name = method + '.dat'
    save_data(file_name, np.mean(opt_values_overall), np.std(opt_values_overall)/np.sqrt(nrep), int(episode == train_episodes), target_model_overall)
#%% oracle clusterings
if method == "oracle_clusterings":
    print('oracle clustering')
    target_model_cluster1, episode_cluster1 = run(States[:int(N/2), :, :], Actions[:int(N/2), :], Rewards[:int(N/2), :])
    print(' target_model_cluster1', target_model_cluster1)
    target_model_cluster2, episode_cluster2 = run(States[int(N/2):, :, :], Actions[int(N/2):, :], Rewards[int(N/2):, :])
    opt_values_cluster = []
    for i in range(nrep):
        if i < nrep/2:
            observation = generate_initial_state()
            encoded_reshaped = observation.reshape([1, observation.shape[0]])
            predicted = target_model_cluster1.predict(encoded_reshaped, verbose=0).flatten()
            opt_values_cluster.append(max(predicted))
        else:
            observation = generate_initial_state()
            encoded_reshaped = observation.reshape([1, observation.shape[0]])
            predicted = target_model_cluster2.predict(encoded_reshaped, verbose=0).flatten()
            opt_values_cluster.append(max(predicted))
    print("oracle cluster value:", np.mean(opt_values_cluster)) #signal0.5: 0.17776427
    file_name = method + '.dat'
    # print('file_name', file_name)
    save_data(file_name, np.mean(opt_values_cluster), np.std(opt_values_cluster)/np.sqrt(nrep), (int(episode_cluster1 == train_episodes)+int(episode_cluster2 == train_episodes))/2, target_model_cluster1, target_model_cluster2)

#%% oracle change points
if method == "oracle_cp":
    States_list = []
    Actions_list = []
    Rewards_list = []
    # for i in range(N):
    #     States_list.append(States[i, int(changepoints_true[i]):, :].reshape((1, -1, p)))
    #     Actions_list.append(Actions[i, int(changepoints_true[i]):].reshape(1,-1))
    #     Rewards_list.append(Rewards[i, int(changepoints_true[i]):].reshape(1,-1))
    for i in range(N):
        States_list.append(States[i, int(changepoints_true[i]):, :])
        Actions_list.append(Actions[i, int(changepoints_true[i]):])
        Rewards_list.append(Rewards[i, int(changepoints_true[i]):])
    
    target_model_cp, episode_cp = run(States_list, Actions_list, Rewards_list)
    opt_values_cp = []
    for i in range(nrep):
            observation = generate_initial_state()
            encoded_reshaped = observation.reshape([1, observation.shape[0]])
            predicted = target_model_cp.predict(encoded_reshaped, verbose=0).flatten()
            opt_values_cp.append(max(predicted))
    print("oracle cp value:", np.mean(opt_values_cp)) #signal0.5: 0.17776427
    file_name = method + '.dat'
    save_data(file_name,np.mean(opt_values_cp), np.std(opt_values_cp)/np.sqrt(nrep), int(episode_cp == train_episodes), target_model_cp)

#%% oracle
if method == "oracle":
    target_model_oracle1, episode_oracle1 = run(States[:int(N/2), int(changepoints_true[0]):, :], Actions[:int(N/2), int(changepoints_true[0]):], Rewards[:int(N/2), int(changepoints_true[0]):])
    target_model_oracle2, episode_oracle2 = run(States[int(N/2):, int(changepoints_true[int(N/2)]):, :], Actions[int(N/2):, int(changepoints_true[int(N/2)]):], Rewards[int(N/2):, int(changepoints_true[int(N/2)]):])
    opt_values_oracle = []
    for i in range(nrep):
        if i < nrep/2:
            observation = generate_initial_state()
            encoded_reshaped = observation.reshape([1, observation.shape[0]])
            predicted = target_model_oracle1.predict(encoded_reshaped, verbose=0).flatten()
            opt_values_oracle.append(max(predicted))
        else:
            observation = generate_initial_state()
            encoded_reshaped = observation.reshape([1, observation.shape[0]])
            predicted = target_model_oracle2.predict(encoded_reshaped, verbose=0).flatten()
            opt_values_oracle.append(max(predicted))
    print("oracle value:", np.mean(opt_values_oracle)) #signal0.5: 0.17776427
    file_name = method + '.dat'
    save_data(file_name,np.mean(opt_values_oracle), np.std(opt_values_oracle)/np.sqrt(nrep), (int(episode_oracle1 == train_episodes)+int(episode_oracle2 == train_episodes))/2, target_model_oracle1, target_model_oracle2)

#%% indi
if method == "indi":
    opt_values_indi = []
    episode_indi = []
    for i in range(N):
        target_model_i, episode_i = run(States[i,:,:].reshape((1,-1,p)), Actions[i, :].reshape((1,-1)), Rewards[i, :].reshape((1,-1)), MIN_REPLAY_SIZE=50)
        episode_indi.append(episode_i)
        for j in range(int(nrep/N)):
            observation = generate_initial_state()
            encoded_reshaped = observation.reshape([1, observation.shape[0]])
            predicted = target_model_i.predict(encoded_reshaped, verbose=0).flatten()
            opt_values_indi.append(max(predicted))
    print("indi value:", np.mean(opt_values_indi)) #signal0.5: 0.17776427
    file_name = method + '.dat'
    save_data(file_name,np.mean(opt_values_oracle), np.std(opt_values_oracle)/np.sqrt(nrep), np.mean(episode_indi), target_model_oracle1, target_model_oracle2)

