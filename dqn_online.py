'''
Simulate nonstationary time series data and apply Q-learning.
Data generation mechanism resembles the IHS data
'''
#!/usr/bin/python
import platform, sys, os, pickle, re
plat = platform.platform()
# print(plat)
# if plat == 'macOS-12.3.1-x86_64-i386-64bit' or plat == 'macOS-10.16-x86_64-i386-64bit': ##local
#     os.chdir("/Users/mengbing/Documents/research/change_point_clustering/HeterRL/simulation_real")
#     sys.path.append("/Users/mengbing/Documents/research/change_point_clustering/HeterRL")
# elif plat == 'Linux-3.10.0-957.el7.x86_64-x86_64-with-centos-7.6.1810-Core': # biostat cluster
#     os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_real")
#     sys.path.append("/home/mengbing/research/RL_nonstationary/code2/")
# elif plat == 'Linux-3.10.0-1160.6.1.el7.x86_64-x86_64-with-glibc2.17' or plat == 'Linux-3.10.0-1160.45.1.el7.x86_64-x86_64-with-glibc2.17':  # greatlakes
#     os.chdir("/home/mengbing/research/RL_nonstationary/code2/simulation_nonstationary_real")
#     sys.path.append("/home/mengbing/research/RL_nonstationary/code2/")
# elif plat == 'Linux-3.10.0-1160.15.2.el7.x86_64-x86_64-with-centos-7.9.2009-Core': #gcp
#     os.chdir("/home/limengbinggz_gmail_com/rl_nonstationary/simulation_nonstationary_real")
#     sys.path.append("/home/limengbinggz_gmail_com/rl_nonstationary/")

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

mag_factor=3
# oracle change points and cluster membership
# g_index_true = np.append(np.append(np.zeros(int(N/2)), np.ones(int(N/2))))
# changepoints_true = np.append(np.append(89*np.ones(int(N/3)), 79*np.ones(int(N/3))), 69*np.ones(int(N/3)))
signal_factor = 0.5

MIN_REPLAY_SIZE = 1000
#%%
import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import time
import random
from scipy.stats import multivariate_normal

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# An episode a full game
train_episodes = 300
test_episodes = 100


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
    print('train')
    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

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
    err_mean = np.zeros(2)
    err_cov = np.diag([cov,cov])
    return transition_matrix[At] @ St + \
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
    replay_memory = deque(maxlen=50_000)

    gamma = 0.9
    steps_to_update_target_model = 0
    cum_rewards= 0.0
    mean_cr_old = 0.0
    mean_cr = 0.0
    num_rewards = 0
    converge=0

    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = generate_initial_state()
        done = False
        while not done:
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

            if steps_to_update_target_model >= 99:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(
                    total_training_rewards, episode, reward))
                total_training_rewards += 1
                # if steps_to_update_target_model >= 100:
                print('Copying main network weights to the target network weights')
                target_model.set_weights(model.get_weights())
                steps_to_update_target_model = 0
                break
            # if the average of the rewards does not change, we stop the online learning
            if np.abs(mean_cr - mean_cr_old) < tol:
                converge=1
                break
            else:
                mean_cr_old = mean_cr
        if converge:
            break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
        
    return target_model,episode


#%%
## cluster 1:
transition_matrix = [None] * 2
transition_matrix[0] = np.array([[0,0],
                                 [0,0]]) + signal_factor * np.array([[0.25,0],
                                                                     [0,-0.25]])
transition_matrix[1] = np.array([[0,0],
                                 [0,0]]) + signal_factor * np.array([[-0.25,0],
                                                                     [0,0.25]])

target_model_1, episode_1 = run(transition_matrix)
# now generate a bunch of initial states and calculate the optimal value
nrep = N*10
opt_values_1 = []
for i in range(nrep):
    observation = generate_initial_state()
    encoded_reshaped = observation.reshape([1, observation.shape[0]])
    predicted = target_model_1.predict(encoded_reshaped, verbose=0).flatten()
    opt_values_1.append(max(predicted))
print("Optimal value:", np.mean(opt_values_1)) #signal0.5: 0.16361411


## cluster 2:
transition_matrix = [None] * 2
transition_matrix[0] = np.array([[0,0],
                                 [0,0]]) + signal_factor * np.array([[-0.25,0],
                                                                     [0,0.25]])
transition_matrix[1] = np.array([[0,0],
                                 [0,0]]) + signal_factor * np.array([[0.25,0],
                                                                     [0,-0.25]])
target_model_2, episode_2 = run(transition_matrix)
# now generate a bunch of initial states and calculate the optimal value
opt_values_2 = []
for i in range(nrep):
    observation = generate_initial_state()
    encoded_reshaped = observation.reshape([1, observation.shape[0]])
    predicted = target_model_2.predict(encoded_reshaped, verbose=0).flatten()
    opt_values_2.append(max(predicted))
print("Optimal value:", np.mean(opt_values_2)) # signal0.5: 0.45594767

