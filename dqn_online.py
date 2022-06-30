'''
Simulate nonstationary time series data and apply Q-learning.
Data generation mechanism resembles the IHS data
'''
import platform, sys, os, re, pickle
import tensorflow as tf
# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)
import numpy as np
plat = platform.platform()
# print(plat)
if plat == 'Windows-10-10.0.14393-SP0': ##local
    os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
    sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
elif plat == 'Linux-5.4.188+-x86_64-with-Ubuntu-18.04-bionic': # colab
    os.chdir('/content/drive/My Drive/Colab Notebooks/heterRL')
    sys.path.append("/content/drive/My Drive/Colab Notebooks/heterRL")
elif plat == 'Linux-3.10.0-1160.6.1.el7.x86_64-x86_64-with-glibc2.17' or plat == 'Linux-3.10.0-1160.45.1.el7.x86_64-x86_64-with-glibc2.17':  # greatlakes
    os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_real")
    sys.path.append("/home/mengbing/research/RL_nonstationary/code2/")
elif plat == 'Linux-3.10.0-1160.15.2.el7.x86_64-x86_64-with-centos-7.9.2009-Core': #gcp
    os.chdir("/home/limengbinggz_gmail_com/rl_nonstationary/simulation_nonstationary_real")
    sys.path.append("/home/limengbinggz_gmail_com/rl_nonstationary/")
import simu.simulate_data_pd as sim
# %% parameters to simulate data
cluster_id = int(sys.argv[1])
print('clusterid',cluster_id)
signal_factor = float(sys.argv[2])
print('signal_factor ',signal_factor )
test_size_factor = int(sys.argv[3])
train_episodes = int(sys.argv[4])
T1_interval = 100
N=10
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
gamma = 0.95
mag_factor=3
# oracle change points and cluster membership
# g_index_true = np.append(np.append(np.zeros(int(N/2)), np.ones(int(N/2))))
# changepoints_true = np.append(np.append(89*np.ones(int(N/3)), 79*np.ones(int(N/3))), 69*np.ones(int(N/3)))
# signal_factor = 1
nrep = N*test_size_factor
MIN_REPLAY_SIZE = 1000
trans_setting='pwconst2'
reward_setting = 'homo'
print(1)
#%% simulate data and value evaluation
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


#%%
from tensorflow import keras

from collections import deque
import random
from scipy.stats import multivariate_normal

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
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
    return model.predict(state.reshape([1, state.shape[0]]),verbose=0)[0]

def train(discount_factor, replay_memory, model, target_model, done):
    learning_rate = 0.7  # Learning rate

    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    # print('train')
    batch_size = 64 * 2
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
    # err_mean = np.zeros(2)
    # err_cov = np.diag([cov,cov])
    # return transition_matrix[At] @ St + \
    #        multivariate_normal.rvs(err_mean, err_cov)
    # St_full = np.insert(St, 0, 1, axis=0)
    St_full = St
    err_mean = np.zeros(2)
    err_cov = np.diag([cov,cov])
    return transition_matrix[At] @ St_full + \
           multivariate_normal.rvs(err_mean, err_cov)


def get_reward(St):
    return St[0]

def run(transition_matrix, tol = 1e-5):
    epsilon = 1  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1  # You can't explore more than 100% of the time
    min_epsilon = 0.01  # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    # 1. Initialize the Target and Main models
    state_space_shape = (2,)
    # Main Model (updated every 4 steps)
    model = agent(state_space_shape, 2)
    # Target Model (updated every 100 steps)
    target_model = agent(state_space_shape, 2)
    target_model.set_weights(model.get_weights())
    target_old = agent(state_space_shape, 2)
    target_old.set_weights(model.get_weights())
    replay_memory = deque(maxlen=50_000)
    S1 = []
    S2= []
    gamma = 0.9
    steps_to_update_target_model = 0
    cum_rewards= 0.0
    mean_cr = 0.0
    mean_cr_old =-100
    num_rewards = 0
    converge=0

    for episode in range(train_episodes):
        print('episode', episode)
        total_training_rewards = 0
        observation = generate_initial_state()
        S1.append(observation[0].item())
        S2.append(observation[1].item())
        done = False
        t=0
        while not done:
            print(t)
            t+=1
            steps_to_update_target_model += 1
            # if True:
            #     env.render()

            random_number = np.random.rand()
            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                action = np.random.binomial(1, 0.25)
            else:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                encoded = observation
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                predicted = model.predict(encoded_reshaped).flatten()
                action = np.argmax(predicted)
            new_observation = transition(observation, action, transition_matrix)
            S1.append(new_observation[0].item())
            S2.append(new_observation[1].item())
            reward = get_reward(new_observation)
            # print('reward', reward)
            cum_rewards += reward
            num_rewards += 1
            mean_cr = cum_rewards/num_rewards
            print('mean_cr',mean_cr)
            # done = steps_to_update_target_model >= 150
            replay_memory.append([observation, action, reward, new_observation, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(gamma, replay_memory, model, target_model, done)

            observation = new_observation
            total_training_rewards += reward
            
            if np.abs(mean_cr - mean_cr_old) < tol:
                converge=1
                break
            else:
                mean_cr_old = mean_cr
                
            if steps_to_update_target_model >= 99 and len(replay_memory) > MIN_REPLAY_SIZE :
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(
                    total_training_rewards, episode, reward))
                total_training_rewards += 1
                # if steps_to_update_target_model >= 100:
                print('Copying main network weights to the target network weights')
                target_old.set_weights(target_model.get_weights())
                target_model.set_weights(model.get_weights())
                steps_to_update_target_model = 0
                # err = 0.0
                # print('len(model.layers)',len(model.layers))
                # for i in range(len(model.layers)):
                #     err += np.linalg.norm(target_model.get_weights()[i]-target_old.get_weights()[i], ord=2)
                # print('err',err)
                # if err < tol:
                #     converge=1
                break
            # if the average of the rewards does not change, we stop the online learning
            
        if converge:
            break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        if divmod(episode, 10)[1] == 0:
            nrep = N*5
            opt_values_1 = []
            for i in range(nrep):
                observation = generate_initial_state()
                encoded_reshaped = observation.reshape([1, observation.shape[0]])
                predicted = target_model.predict(encoded_reshaped, verbose=0).flatten()
                opt_values_1.append(max(predicted))
            print("Optimal value:", np.mean(opt_values_1)) #signal0.5: 0.16361411

    return target_model,episode
#%% save data
print('signal_factor',signal_factor, ' signal_factor*-0.25', signal_factor*-0.25)
coef1=[[signal_factor*0, signal_factor*0, signal_factor*-0.25],[signal_factor*0, signal_factor*0, signal_factor*0.25]]
print('coef',coef1)
def setpath(cluster_id):
    # os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
    append_name = '_N' + str(N) + '_1d'
    if not os.path.exists('results/value_evaluation'):
        os.makedirs('results/value_evaluation', exist_ok=True)
    data_path = 'results/value_evaluation/coef'+re.sub("\\ ", "",re.sub("\\.", "", re.sub("\\]","", re.sub("\\[", "", re.sub("\\, ", "", str(coef1))))))+'/sim_result_online'+ append_name
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    data_path += '/sim_clusterid' + str(cluster_id)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    os.chdir(data_path)
    return
# setpath(seed)

def save_data(file_name, estimated_value,std_value,converge,model):
    setpath(cluster_id)
    with open(file_name, "wb")as f:
        pickle.dump({'value':estimated_value,
                      'std':std_value,
                     # 'path':estimated_value_path,
                      'converge':converge,
                     'weights':[model.get_weights()]},f)
    
    # model.save('model')
    return

# direct the screen output to a file
# stdoutOrigin = sys.stdout
# sys.stdout = open("results/value_evaluation/log_clusterid" +str(cluster_id)+'_signal' + str(signal_factor)+ ".txt", "w")
# print("\nName of Python script:", sys.argv[0], 'train episode:', train_episodes)
# sys.stdout.flush()

#%% cluster 0:
if cluster_id == 0:
    print(cluster_id)
    transition_matrix = [None] * 2
    transition_matrix[0] = np.array([[0,0],
                                     [0,0]]) + signal_factor * np.array([[0.25,0],
                                                                         [0,-0.25]])
    transition_matrix[1] = np.array([[0,0],
                                     [0,0]]) + signal_factor * np.array([[-0.25,0],
                                                                         [0,0.25]])
                                            
    target_model_1, episode_1 = run(transition_matrix)
    # now generate a bunch of initial states and calculate the optimal value
    # opt_values_1 = []
    # for i in range(nrep):
    #     observation = generate_initial_state()
    #     encoded_reshaped = observation.reshape([1, observation.shape[0]])
    #     predicted = target_model_1.predict(encoded_reshaped, verbose=0).flatten()
    #     opt_values_1.append(max(predicted))
    opt_values_1 = calculate_value(transition_matrix, target_model_1, transition_matrix, nrep, T1_interval)
    print("Optimal value:", np.mean(opt_values_1),'episode', episode_1) #signal0.5: 0.16361411
    file_name = 'clusterid'+str(cluster_id) + '.dat'
    save_data(file_name, np.mean(opt_values_1), np.std(opt_values_1)/np.sqrt(nrep), int(episode_1 == train_episodes), target_model_1)

#%% cluster 1:
if cluster_id == 1:
    print(cluster_id)
    transition_matrix = [None] * 2
    transition_matrix[0] = np.array([[0,0],
                                     [0,0]]) + signal_factor * np.array([[-0.25,0],
                                                                         [0,0.25]])
    transition_matrix[1] = np.array([[0,0],
                                     [0,0]]) + signal_factor * np.array([[0.25,0],
                                                                         [0,-0.25]])
    target_model_2, episode_2 = run(transition_matrix)
    # now generate a bunch of initial states and calculate the optimal value
    # opt_values_2 = []
    # for i in range(nrep):
    #     observation = generate_initial_state()
    #     encoded_reshaped = observation.reshape([1, observation.shape[0]])
    #     predicted = target_model_2.predict(encoded_reshaped, verbose=0).flatten()
    #     opt_values_2.append(max(predicted))
    opt_values_2 = calculate_value(transition_matrix, target_model_2, transition_matrix, nrep, T1_interval)
    print("Optimal value:", np.mean(opt_values_2), 'episode', episode_2) # signal0.5: 0.45594767
    file_name = 'clusterid'+str(cluster_id) + '.dat'
    save_data(file_name, np.mean(opt_values_2), np.std(opt_values_2)/np.sqrt(nrep), not int(episode_2 == train_episodes-1), target_model_2)
