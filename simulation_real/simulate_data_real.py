'''
Simulate stationary time series data and apply Q-learning.
Simulate real data with 4-dimensional states
'''

import numpy as np
from scipy.stats import multivariate_normal
from copy import deepcopy


#%%
def simulate(system_settings, seed=0, S0 = None, T0=0, T_total=26, burnin=0,
             optimal_policy_model=None, epsilon_greedy = 0.05,
             mean0 = 0.0, cov0 = 0.5, mean = 0.0, cov = 0.25, normalized=[0.0, 1.0]):
    '''
    simulate states, rewards, and action data
    :param mean0: mean vector for the initial state S_0
    :param cov0: covariance matrix for the initial state S_0
    :param seed: numpy random seed
    :param S0: initial states of N subjects
    :param A0: initial actions of N subjects
    :param optimal_policy_model: input actions and simulate state and rewards following the input actions. Can be used for
    Monte Carlo policy evaluation
    :return: a list [ states, rewards, actions]
    '''
    # number of total time points
    T = system_settings['T']
    # number of individuals
    N = system_settings['N']
    change_points = deepcopy(system_settings['changepoints'])

    # residual mean and covariance when generating new states
    mean = np.zeros(3)
    cov = np.diag([1, 1, 0.2])

    # set seed
    np.random.seed(seed)
    States = np.zeros([N, T + 1, 3])  # S[i][t]
    # print(States.shape)
    Rewards = np.zeros([N, T])  # R[i,t]
    Actions = np.zeros([N, T])  # Actions[i,t]

    change_points.append(T)

    # print(np.random.normal(mean0, cov0, 1))

    #%% random actions
    if optimal_policy_model is None:

        # generate initial state S_0 and action A_0
        # States[:, 0, 0] = np.random.normal(mean0, cov0, N)
        # Actions[:, 0] = np.random.binomial(1, 0.5, N)

        t = 0
        for i in range(N):  # for each individual
            # generate initial state S_0 and action A_0
            if S0 is None:
                St = np.ones(3)
                St[0] = np.random.normal(20, 3)  # step count of week t
                St[1] = np.random.normal(20, 2)  # sleep of week t
                St[2] = np.random.normal(7, 1)  # mood score of week t
            else:
                St = S0 #[i, :, ]

            # St = np.random.normal(mean0, cov0, 1)
            # print("St =", St)
            States[i, 0, :] = St
            At = np.random.binomial(1, 0.5, 1)[0]
            Actions[i, t] = At
            Rewards[i, t] = system_settings['reward_functions'](St, At, t)
            # generate S_i,t+1
            St = system_settings['state_functions'][0](St, (2.0 * At - 1.0), t) + multivariate_normal.rvs(mean, cov, 1)
            # print("St =", St)
            States[i, t + 1, :] = St
            # print("States[i, t + 1, :] =", States[i, t + 1, :])

        # for the subsequent time points
        for i in range(N):  # each individual i
            St = States[i, 1, :]
            previous_change_point = 1
            for segment in range(len(change_points)):  # for each change point
                # print('segment', segment)
                for t in range(previous_change_point, change_points[segment]):
                    # if i == 1:
                    #     print('t =', t, ', sgement =', segment)
                    # print('t', t, ', previous_change_point',previous_change_point, ', change_points[segment]',change_points[segment])
                    ## generate action
                    At = np.random.binomial(1, 0.5, 1)[0]
                    At = int(At)
                    Actions[i, t] = At

                    ## generate immediate response R_i,t
                    Rt = system_settings['reward_functions'](St, At, t)
                    Rewards[i, t] = Rt

                    ## compute the next state
                    St = system_settings['state_functions'][segment](St, (2.0 * At - 1.0), t) + multivariate_normal.rvs(mean, cov, 1)[0]
                    States[i, t + 1, :] = St
                    # print("States[i, t + 1, 0] =", States[i, t + 1, 0])

                previous_change_point = change_points[segment]

    #%% with known policy
    else:
        myState = np.zeros([1, 2, 3])
        t = 0
        for i in range(N):  # for each individual
            # print("i =", i, ", t = 0")
            # generate initial state S_0 and action A_0
            if S0 is None:
                St = np.ones(3)
                St[0] = np.random.normal(20, 3)  # step count of week t
                St[1] = np.random.normal(20, 2)  # sleep of week t
                St[2] = np.random.normal(7, 1)  # mood score of week t
            else:
                St = S0 #[i, :, ]
            # print("St =", St)
            # print("St =", St)
            States[i, 0, :] = St
            # compute the current action
            myState[0, 0, :] = St
            # generate policy
            if np.random.rand() < epsilon_greedy:
                At = np.random.binomial(1, 0.5, 1)[0]
                # print("epsilon At =", At)
            else:
                At = optimal_policy_model.predict(myState).opt_action[0]
                # print("optimal At =", At)
            At = int(At)
            Actions[i, t] = At
            Rewards[i, t] = system_settings['reward_functions'](St, At, t)
            # generate S_i,t+1
            St = system_settings['state_functions'][0](St, (2.0 * At - 1.0), t) + multivariate_normal.rvs(mean, cov, 1)[0]
            # print("St =", St)
            States[i, t + 1, :] = St
            # print("States[i, t + 1, :] =", States[i, t + 1, :])

        # for the subsequent time points
        for i in range(N):  # each individual i
            St = States[i, 1, :]
            previous_change_point = 1
            # print('len(change_points)',len(change_points), 'change_points',change_points)
            for segment in range(len(change_points)):  # for each change point
                for t in range(previous_change_point, change_points[segment]):
                    # print('t', t,', previous_change_point',previous_change_point,', change_points[',segment,']',change_points[segment])
                    ## generate action
                    # compute the current action
                    myState[0, 0, :] = St
                    # generate policy
                    if np.random.rand() < epsilon_greedy:
                        At = np.random.binomial(1, 0.5, 1)[0]
                        # print("i =", i, ", t =", t, "At =", At)
                    else:
                        # print("myState =", myState[:,:,0])
                        # print("action =", optimal_policy_model.predict(myState).opt_action[0])
                        At = optimal_policy_model.predict(myState).opt_action[0]
                    # print("i =", i, ", t =", t, "At =", At)
                    At = int(At)
                    Actions[i, t] = At

                    ## generate immediate response R_i,t
                    Rt = system_settings['reward_functions'](St, At, t)
                    Rewards[i, t] = Rt

                    ## compute the next state
                    St = system_settings['state_functions'][segment](St, (2.0 * At - 1.0), t) + multivariate_normal.rvs(mean, cov, 1)[0]
                    States[i, t + 1, :] = St
                    # print("States[i, t + 1, 0] =", States[i, t + 1, 0])

                previous_change_point = change_points[segment]

    # convert Actions to integers
    Actions = Actions.astype(int)
    return States, Rewards, Actions

