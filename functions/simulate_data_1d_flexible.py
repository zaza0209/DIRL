'''
Simulate stationary time series data and apply Q-learning.
Generate 1 dimensional states
'''


import numpy as np
import math
from copy import deepcopy
from copy import copy



def psi(t, w):
    if t <= 0:
        # print("NULL")
        return 0
    else:
        return math.exp(-w / t)

def smooth_transform(x, f1, f2, x0, x1, w=0.01):
    '''
    Define a smooth transformation function on interval [x0, x1], connecting smoothly function f1 on x < x0 and f2 on x > x1
    :param x: the transformed function value to be evaluated at
    :return: a scalar of transformed function value
    '''

    if x > x0:
        ratio = (x - x0) / (x1 - x0)
        return f1(x) + (f2(x) - f1(x)) / (psi(ratio, w) + psi(1 - ratio, w)) * psi(ratio, w)
    elif x <= x0:
        return f1(x)
    elif x >= x1:
        return f2(x)




#%%
def simulate(system_settings, seed=None, S0 = None, T0=0, T_total=26, burnin=0,
             optimal_policy_model=None, epsilon_greedy = 0.05,
             mean0 = 0.0, cov0 = 0.5, normalized=[0.0, 1.0]):
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
    p = system_settings.get('p', 1)
    
    change_points = deepcopy(system_settings['changepoints'])

    np.random.seed(seed)
    States = np.zeros([N, T + 1, p])  # S[i][t]
    Rewards = np.zeros([N, T])  # R[i,t]
    Actions = np.zeros([N, T])  # Actions[i,t]

    if system_settings['state_change_type'] == 'smooth' or system_settings['reward_change_type'] == 'smooth':
        deltaT = system_settings['delta'] * T
    change_points.append(T)


    #%% random actions
    if optimal_policy_model is None:

        t = 0
        for i in range(N):  # for each individual
            if S0 is None:
                if p == 1:
                    St = np.random.normal(mean0, cov0, 1)
                else:
                    St = np.random.multivariate_normal(mean0, cov0)
            else:
                St = S0[i, :]
                
            count = 0
            while count < burnin:
                States[i, 0, :] = St
                At = np.random.binomial(1, 0.5, 1)[0]
                Actions[i, t] = At
                # print('St', St, ' t+count',  t+count)
                Rewards[i, t] = system_settings['reward_functions'][0](St, At, t=t+count)
                St = system_settings['state_functions'][0](St, At, t=t+count) 
                count += 1
            States[i, t + 1, :] = St

            # # St = np.random.normal(mean0, cov0, 1)
            # States[i, 0, :] = St
            # At = np.random.binomial(1, 0.5, 1)[0]
            # Actions[i, t] = At
            # Rewards[i, t] = system_settings['reward_functions'][0](St, At, t)
            # # generate S_i,t+1
            # St = system_settings['state_functions'][0](St, At, t) #+ np.random.normal(mean, cov, 1)[0]
            # # print("St =", St)
            # States[i, t + 1, :] = St
            # print("States[i, t + 1, :] =", States[i, t + 1, :])

        # for the subsequent time points
        for i in range(N):  # each individual i
            St = States[i, 1, :]
            previous_change_point = 1
            for segment in range(len(change_points)):  # for each change point
                # print('segment', segment)
                for t in range(previous_change_point, change_points[segment]):
                    # print('t', t, ', previous_change_point',previous_change_point, ', change_points[segment]',change_points[segment])
                    ## generate action
                    At = np.random.binomial(1, 0.5, 1)[0]
                    At = int(At)
                    Actions[i, t] = At

                    ## generate immediate response R_i,t
                    if system_settings['reward_change_type'] == 'pwconst2':
                        Rt = system_settings['reward_functions'][segment](St, At, t)
                        # print("t =", t_current, " Reward: pwconst, segment", segment)
                    elif system_settings['reward_change_type'] == 'homogeneous':
                        Rt = system_settings['reward_functions'][0](St, At, t)
                    elif system_settings['reward_change_type'] == 'smooth':
                        if segment < len(change_points)-1: # before reaching the last segment
                            # before the smooth change point: simply use the constant part
                            if t <= change_points[segment] - deltaT - 1:
                                Rt = system_settings['reward_functions'][segment](St, At, t)
                                # print("t =", t_current, ": smooth; before cp")
                            else: # during the smooth change
                                def f1(tt):
                                    return system_settings['reward_functions'][segment](St, At, tt)
                                def f2(tt):
                                    return system_settings['reward_functions'][segment + 1](St, At, tt)
                                Rt = smooth_transform(t, f1, f2, change_points[segment] - deltaT, change_points[segment])
                        elif segment == len(change_points) - 1:  # at the last segment
                            Rt = system_settings['reward_functions'][segment](St, At, t)
                    Rewards[i, t] = Rt

                    ## compute the next state
                    if system_settings['state_change_type'] == 'pwconst2':
                        St = system_settings['state_functions'][segment](St, At, t) #+ np.random.normal(mean, cov, 1)[0]
                    elif system_settings['state_change_type'] == 'homogeneous':
                        St = system_settings['state_functions'][0](St, At, t)  #+ np.random.normal(mean, cov, 1)[0]
                    elif system_settings['state_change_type'] == 'smooth':
                        if segment < len(change_points)-1: # before reaching the last segment
                            # before the smooth change point: simply use the constant part
                            if t <= change_points[segment] - deltaT:
                                St = system_settings['state_functions'][segment](St, At, t)  #+ np.random.normal(mean, cov, 1)[0]
                            else: # during the smooth change

                                def f1(tt):
                                    return system_settings['state_functions'][segment](St, At, tt)
                                def f2(tt):
                                    return system_settings['state_functions'][segment + 1](St, At, tt)
                                St = smooth_transform(t, f1, f2, change_points[segment] - deltaT, change_points[segment])  #+ np.random.normal(mean, cov, 1)[0]
                        elif segment == len(change_points) - 1:  # at the last segment
                            St = system_settings['state_functions'][segment](St, At, t)  #+ np.random.normal(mean, cov, 1)[0]
                    States[i, t + 1, :] = St

                previous_change_point = change_points[segment]

    #%% with known policy
    else:
        myState = np.zeros([2, 2, p])
        t = 0
        for i in range(N):  # for each individual
            if S0 is None:
                if p ==1:
                    St = np.random.normal(mean0, cov0, 1)
                else:
                    St = np.random.multivariate_normal(mean0, cov0)
            else:
                St = S0[i, :]
            
            count = 0
            while count < burnin:
                States[i, 0, :] = St
                myState[0, 0, :] = St
                # generate policy
                if np.random.rand() < epsilon_greedy:
                    At = np.random.binomial(1, 0.5, 1)[0]
                else:
                    At =  optimal_policy_model.predict(myState).opt_action[0] #target_policy(St.reshape(1, 1, -1))
                At = int(At)
                Actions[i, t] = At
                Rewards[i, t] = system_settings['reward_functions'][0](St, At, t=t+count)
                St = system_settings['state_functions'][0](St, At,t= t+count)# np.random.normal(mean, cov, 1)[0]
                count +=1 
            States[i, t + 1, :] = St
            
            # States[i, 0, :] = St
            # # compute the current action
            # myState[0, 0, 0] = St
            # # generate policy
            # if np.random.rand() < epsilon_greedy:
            #     At = np.random.binomial(1, 0.5, 1)[0]
            #     # print("epsilon At =", At)
            # else:
            #     At = optimal_policy_model.predict(myState).opt_action[0]
            #     # print("optimal At =", At)
            # At = int(At)
            # Actions[i, t] = At
            # Rewards[i, t] = system_settings['reward_functions'][0](St, At, t)
            # # generate S_i,t+1
            # St = system_settings['state_functions'][0](St, At, t) #+ np.random.normal(mean, cov, 1)[0]
            # States[i, t + 1, :] = St

        # for the subsequent time points
        for i in range(N):  # each individual i
            St = States[i, 1, :]
            previous_change_point = 1
            for segment in range(len(change_points)):  # for each change point
                for t in range(previous_change_point, change_points[segment]):
                    ## generate action
                    # compute the current action
                    myState[0, 0, :] = St
                    # generate policy
                    if np.random.rand() < epsilon_greedy:
                        At = np.random.binomial(1, 0.5, 1)[0]
                    else:
                        At = optimal_policy_model.predict(myState).opt_action[0]
                    At = int(At)
                    Actions[i, t] = At

                    ## generate immediate response R_i,t
                    if system_settings['reward_change_type'] == 'pwconst2':
                        Rt = system_settings['reward_functions'][segment](St, At, t)
                    elif system_settings['reward_change_type'] == 'homogeneous':
                        Rt = system_settings['reward_functions'][0](St, At, t)
                    elif system_settings['reward_change_type'] == 'smooth':
                        if segment < len(change_points)-1: # before reaching the last segment
                            # before the smooth change point: simply use the constant part
                            if t <= change_points[segment] - deltaT - 1:
                                Rt = system_settings['reward_functions'][segment](St, At, t)
                                # print("t =", t_current, ": smooth; before cp")
                            else: # during the smooth change
                                def f1(tt):
                                    return system_settings['reward_functions'][segment](St, At, tt)
                                def f2(tt):
                                    return system_settings['reward_functions'][segment + 1](St, At, tt)
                                Rt = smooth_transform(t, f1, f2, change_points[segment] - deltaT, change_points[segment])
                                # print("t =", t_current, ": smooth; during change")
                        elif segment == len(change_points) - 1:  # at the last segment
                            Rt = system_settings['reward_functions'][segment](St, At, t)
                            # print("t =", t_current, ": smooth; last segment")
                    Rewards[i, t] = Rt

                    ## compute the next state
                    if system_settings['state_change_type'] == 'pwconst2':
                        St = system_settings['state_functions'][segment](St, At, t) #+ np.random.normal(mean, cov, 1)[0]
                    elif system_settings['state_change_type'] == 'homogeneous':
                        St = system_settings['state_functions'][0](St, At, t) #+ np.random.normal(mean, cov, 1)[0]
                    elif system_settings['state_change_type'] == 'smooth':
                        if segment < len(change_points)-1: # before reaching the last segment
                            # before the smooth change point: simply use the constant part
                            if t <= change_points[segment] - deltaT:
                                St = system_settings['state_functions'][segment](St, At, t) #+ np.random.normal(mean, cov, 1)[0]
                            else: # during the smooth change

                                def f1(tt):
                                    return system_settings['state_functions'][segment](St, At, tt)
                                def f2(tt):
                                    return system_settings['state_functions'][segment + 1](St, At, tt)
                                St = smooth_transform(t, f1, f2, change_points[segment] - deltaT, change_points[segment]) #+ np.random.normal(mean, cov, 1)[0]
                        elif segment == len(change_points) - 1:  # at the last segment
                            St = system_settings['state_functions'][segment](St, At, t) # + np.random.normal(mean, cov, 1)[0]
                    States[i, t + 1, :] = St

                previous_change_point = change_points[segment]

    # convert Actions to integers
    Actions = Actions.astype(int)
    return States, Rewards, Actions



