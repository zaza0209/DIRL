import platform, sys, os, re, pickle
import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import time
import random
from scipy.stats import multivariate_normal
os.chdir("C:/Users/test/Dropbox/tml/IHS/simu")
# sys.path.append("/Users/mengbing/Documents/research/RL_nonstationary/cumsumrl/")
sys.path.append("C:/Users/test/Dropbox/tml/IHS/simu") 
import simu.simulate_data_pd as sim
RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# An episode a full game
train_episodes = 300
test_episodes = 100
#%%
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
#%%
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

def train(discount_factor, replay_memory, model, target_model):
    learning_rate = 0.7  # Learning rate
    # MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    print('train')
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


#%%
def run(States, Actions, Rewards, tol = 1e-5):
    N = States.shape[0]

    # 1. Initialize the Target and Main models
    state_space_shape = (2,)
    # Main Model (updated every 4 steps)
    model = agent(state_space_shape, 2)
    # Target Model (updated every 100 steps)
    target_model = agent(state_space_shape, 2)
    target_model.set_weights(model.get_weights())
    target_old = agent(state_space_shape, 2)
    replay_memory = deque(maxlen=50_000)

    steps_to_update_target_model = 0
    cum_rewards= 0.0
    converge=0
    
    for episode in range(train_episodes):
        steps_to_update_target_model=0
        for i in range(N):
            T = Rewards[i,:].shape[0]
            for t in range(T):
                # print('t',t)
                steps_to_update_target_model += 1
                sars1 = []
                sars1.append(States[i,t,:])
                sars1.append(Actions[i,t])
                sars1.append(Rewards[i,t])
                sars1.append(States[i,t+1,:])
                sars1.append(int(t==T-1))
                replay_memory.append(sars1)
                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 4 == 0:
                    train(gamma, replay_memory, model, target_model)
                
                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_old.set_weights(target_model.get_weights())
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                    err = 0.0
                    for i in range(len(model.layers)):
                        err += np.linalg.norm(target_model.get_weights()[i]-target_old.get_weights()[i], ord=2)
                    print(err)
                    if err < tol:
                        converge=1
                        break
            if converge:
                break
        if converge:
            break
        print('Total training rewards: {} after n steps = {} with final reward = {}'.format(
            cum_rewards, episode, reward))
    return target_model,episode

# %% parameters to simulate data
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
mag_factor=3
gamma = 0.9
MIN_REPLAY_SIZE = 300
coef1=[[mag_factor*0, mag_factor*0, mag_factor*-0.25],[mag_factor*0, mag_factor*0, mag_factor*0.25]]
coef =[coef1,coef1[::-1]]# this is acceptable 0609
signal = [[mag_factor*0, mag_factor*0], [mag_factor*0, mag_factor*0]]
seed=0
#%%
def gen_dat(N, T, coef, signal, changepoint_list=None, seed=1):
    np.random.seed(seed)
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
    def transform(x):
        return (x - np.mean(x)) / np.std(x)
    for i in range(p):
        States[:,:,i] = transform(States[:,:,i])
    g_index_true = np.hstack([np.zeros(int(N/2)), np.ones(int(N/2))])
    Actions = Actions.astype(int)
    g_index_true = g_index_true.astype(int)
    return States, Rewards, Actions, changepoints_true, g_index_true

States, Rewards, Actions, changepoints_true, g_index_true = gen_dat(N, T, 
                                                      coef, signal,None,seed + 100)

