'''
Simulate stationary time series data and apply Q-learning.
Simulate real data with 4-dimensional states
'''

import numpy as np
from scipy.stats import multivariate_normal
import copy

# %%
class simulate_data():
    def __init__(self, T, cluster_settings):
        '''
        # :param N:
        :param T: number of time points
        :param clusters: a list or array of length K, where the k-th element is a list of two elements:
            [size_of_cluster_k, true_change_point_of_cluster_k]
        '''
        # clusters = np.array([[30, 10],
        #                      [50, 15],
        #                      [20, 20]])
        self.T = T
        self.K = len(cluster_settings)
        # number of individuals in each cluster
        self.cluster_sizes = [cluster_settings[x]['n'] for x in cluster_settings] #clusters[:, 0]
        self.change_points = copy.deepcopy([cluster_settings[x]['changepoints'] for x in cluster_settings])
        max_time_segment = max([len(x) for x in self.change_points]) + 1
        # print("max_time_segment=", max_time_segment)
        # number of trajectories / episodes
        self.N = sum(self.cluster_sizes)

        # self.state_coefficients = transition_coefficients['state']
        # self.action_coefficients = transition_coefficients['action']
        self.transition_coefficients = [cluster_settings[x]['state'] for x in cluster_settings]
        self.action_coefficients = [cluster_settings[x]['action'] for x in cluster_settings]
        self.transition_matrix = [None] * max_time_segment # for each piece between two change points
        for xx in range(max_time_segment):
            self.transition_matrix[xx] = [None] * 2 #* max_time_segment # action 0 and 1
            # self.transition_matrix[xx][1] = [None] #* max_time_segment # action 0 and 1
        # initialize A_t
        self.At = 1
        # initialize S_t
        self.St = 1
        # initialize S_t-1
        self.Stm1 = 1
        self.err_mean = np.zeros(3)
        self.err_cov = np.diag([1, 1, 0.2])



    def transition(self, k, t, time_segment):
        '''

        :param k: index of cluster k
        :param t: time point t
        :return:
        '''
        St_full = np.insert(self.St, 0, 1, axis=0)
        return self.transition_matrix[time_segment][self.At] @ St_full + \
               multivariate_normal.rvs(self.err_mean, self.err_cov)
        # if t < self.change_points[k-1]:
        #     # print('transition_matrix = transition_matrix[0][self.At]')
        #     return self.transition_matrix[0][self.At] @ St_full + \
        #            multivariate_normal.rvs(self.err_mean, self.err_cov)
        # elif t >= self.change_points[k-1]:
        #     # print('transition_matrix = transition_matrix[1][self.At]')
        #     return self.transition_matrix[1][self.At] @ St_full + \
        #            multivariate_normal.rvs(self.err_mean, self.err_cov)



    def simulate(self, seed=0, T0=0, T_total=26, burnin=0,
                 optimal_policy_model=None, normalized=[0.0, 1.0]):
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
        # if T1 > self.T:
        #     print("error: terminal time T1 of the simulated trajectory should be no more than the total number of time points T")
        #     return 0

        # set seed
        np.random.seed(seed)
        States = np.zeros([self.N, T_total + 1, 3])  # S[i][t]
        # Rewards = np.zeros([self.N, T1-T0])  # R[i,t]
        Actions = np.zeros([self.N, T_total])  # Actions[i,t]

        if optimal_policy_model is None:
            cum_i = 0
            for k in range(self.K): # for the k-th cluster
                Ni = self.cluster_sizes[k] # get cluster size
                cluster_change_points = self.change_points[k] # get change points
                cluster_change_points.append(self.T)
                # pre-calculate two transition matrices for cluster k
                for xx in range(len(cluster_change_points)):
                    # action 0
                    self.transition_matrix[xx][0] = self.transition_coefficients[k][xx] - self.action_coefficients[k][xx]
                    # action 1
                    self.transition_matrix[xx][1] = self.transition_coefficients[k][xx] + self.action_coefficients[k][xx]

                for i in range(Ni): # for each individual
                    ## initial
                    # generate initial state S_i0
                    self.St = np.ones(3)
                    self.St[0] = np.random.normal(20, 3)  # step count of week t
                    self.St[1] = np.random.normal(20, 2)  # sleep of week t
                    self.St[2] = np.random.normal(7, 1)  # mood score of week t
                    if burnin == 0:
                        States[cum_i, 0, :] = self.St
                    # generate policy
                    self.At = np.random.binomial(1, 0.25)

                    ## for the subsequent time points
                    t_recorded = 0
                    time_segment = 0
                    for t in range(-burnin, T_total): # self.change_pt - T0
                        if t >= 0:
                            States[cum_i, t_recorded, :] = self.St
                            Actions[cum_i, t_recorded] = self.At
                            t_recorded += 1
                        if t >= cluster_change_points[time_segment]:
                            time_segment += 1
                            # print("t =", t, "time_segment =", time_segment)
                        # generate S_i,t+1
                        self.St = self.transition(k, t, time_segment)
                        # generate policy
                        self.At = np.random.binomial(1, 0.25)
                        # if i == 0:
                        #     print('k =', k, ', At =', self.At)
                    States[cum_i, t_recorded, :] = self.St
                    cum_i += 1

        #%% if optimal policy is input
        else:
            cum_i = 0
            myState = np.zeros([1, 2, 3])
            for k in range(self.K):  # for the k-th cluster
                Ni = self.cluster_sizes[k]  # get cluster size
                cluster_change_points = self.change_points[k]  # get change points
                cluster_change_points.append(T0+T_total)
                # pre-calculate two transition matrices for cluster k
                for xx in range(len(cluster_change_points)):
                    # action 0
                    self.transition_matrix[xx][0] = self.transition_coefficients[k][xx] - \
                                                    self.action_coefficients[k][xx]
                    # action 1
                    self.transition_matrix[xx][1] = self.transition_coefficients[k][xx] + \
                                                    self.action_coefficients[k][xx]

                for i in range(Ni):  # for each individual
                    # print("i =", i)
                    ## initial
                    # generate initial state S_i0
                    self.St = np.ones(3)
                    self.St[0] = np.random.normal(20, 3)  # step count of week t
                    self.St[1] = np.random.normal(20, 2)  # sleep of week t
                    self.St[2] = np.random.normal(7, 1)  # mood score of week t
                    # print("self.St_0 =", self.St)
                    if burnin == 0:
                        States[cum_i, 0, :] = self.St
                    myState[0, 0, :] = self.St
                    myState[0, 0, :] -= normalized[0]
                    myState[0, 0, :] /= normalized[1]
                    # generate policy
                    self.At = optimal_policy_model.predict(myState).opt_action[0]

                    ## for the subsequent time points
                    t_recorded = 0
                    # find out which segment to start from

                    time_segment = min([i for i,v in enumerate([0] + cluster_change_points) if v >= T0])
                    ## burn-in
                    for t in range(burnin):
                        # generate S_i,t+1
                        self.St = self.transition(k, t, time_segment)
                        # generate policy
                        myState[0, 0, :] = self.St
                        myState[0, 0, :] -= normalized[0]
                        myState[0, 0, :] /= normalized[1]
                        # generate policy
                        self.At = optimal_policy_model.predict(myState).opt_action[0]

                    # after burn-in
                    for t in range(T0, T0+T_total):  # self.change_pt - T0
                        States[cum_i, t_recorded, :] = self.St
                        Actions[cum_i, t_recorded] = self.At
                        t_recorded += 1
                        # print("t =", t, ", time_segment =", time_segment)
                        if t > cluster_change_points[time_segment]:
                            time_segment += 1
                            # print("t =", t, "time_segment =", time_segment)
                        # print("t =", t, ", time_segment =", time_segment)
                        # generate S_i,t+1
                        self.St = self.transition(k, t, time_segment)
                        # generate policy
                        myState[0, 0, :] = self.St
                        myState[0, 0, :] -= normalized[0]
                        myState[0, 0, :] /= normalized[1]
                        # generate policy
                        self.At = optimal_policy_model.predict(myState).opt_action[0]
                        # if i == 0:
                        #     print('k =', k, ', At =', self.At)
                    States[cum_i, t_recorded, :] = self.St
                    cum_i += 1


            # else:
            #     myState = np.zeros([1, 2, 3])
            #     cum_i = 0
            #     for k in range(1, self.K + 1):  # for the k-th cluster
            #         Ni = self.cluster_sizes[k - 1]  # get cluster size
            #         # pre-calculate two transition matrices for cluster k
            #         self.transition_matrix[0][0] = self.transition_coefficients['cluster' + str(k)]['state'] - \
            #                                        self.transition_coefficients['cluster' + str(k)]['action']
            #         self.transition_matrix[0][1] = self.transition_coefficients['cluster' + str(k)]['state'] + \
            #                                        self.transition_coefficients['cluster' + str(k)]['action']
            #         self.transition_matrix[1][0] = self.transition_coefficients['cluster' + str(k)]['state'] + \
            #                                        self.transition_coefficients['cluster' + str(k)]['action']
            #         self.transition_matrix[1][1] = self.transition_coefficients['cluster' + str(k)]['state'] - \
            #                                        self.transition_coefficients['cluster' + str(k)]['action']
            #
            #         for i in range(self.N):  # for each individual
            #             ## initial t=0
            #             # generate initial state S_i0
            #             self.St = np.ones(3)
            #             self.St[0] = np.random.normal(20, 3) # step count of week t
            #             self.St[1] = np.random.normal(20, 2) # sleep of week t
            #             self.St[2] = np.random.normal(7, 1) # mood score of week t
            #             if burnin == 0:
            #                 States[i, 0, :] = self.St
            #             myState[0, 0, :] = self.St
            #             myState[0, 0, :] -= normalized[0]
            #             myState[0, 0, :] /= normalized[1]
            #             # generate policy
            #             self.At = optimal_policy_model.predict(myState).opt_action[0]
            #             # self.At = 1
            #
            #             ## for the subsequent time points
            #             for t in range(-burnin, 0): # self.change_pt - T0
            #                 # generate S_i,t+1
            #                 self.St = self.transition(k, t)
            #                 myState[0, 0, :] = self.St
            #                 myState[0, 0, :] -= normalized[0]
            #                 myState[0, 0, :] /= normalized[1]
            #                 # generate policy
            #                 self.At = optimal_policy_model.predict(myState).opt_action[0]
            #
            #             t_recorded = 0
            #             for t in range(T0[k], T0[k]+T_total): # self.change_pt - T0
            #                 States[cum_i, t_recorded, :] = self.St
            #                 Actions[cum_i, t_recorded] = self.At
            #                 t_recorded += 1
            #                 # generate S_i,t+1
            #                 self.St = self.transition(k, t)
            #                 myState[0, 0, :] = self.St
            #                 myState[0, 0, :] -= normalized[0]
            #                 myState[0, 0, :] /= normalized[1]
            #                 # generate policy
            #                 self.At = optimal_policy_model.predict(myState).opt_action[0]
            #                 # if i == 0:
            #                 #     print('k =', k, ', At =', self.At)
            #             States[cum_i, t_recorded, :] = self.St
            #             cum_i += 1

        # self.change_pt + 2

        # make rewards
        # print(States)
        Rewards = copy.copy(States[:,:-1,0])
        # convert Actions to integers
        Actions = Actions.astype(int)
        return States, Rewards, Actions









    # #%%
    #
    # def transition1(self, t):
    #     '''
    #     Generate transition before change point
    #     '''
    #     At1 = 2 * self.At - 1
    #     # print('t =', t, ", first")
    #     transition_matrix = np.array([[10 + 0.6 * At1, 0.4 + 0.3 * At1, -0.04, 0.1, 0.1 - 0.1*At1],
    #                                   [11 - 0.4 * At1, 0.05, 0.4, 0, 0],
    #                                   [1.2 - 0.5 * At1, -0.02, 0.03 + 0.03*At1, 0.8, 0]])#.03 + 0.01*At1
    #     St_full = np.insert(self.St, 3, self.Stm1, axis=0)
    #     St_full = np.insert(St_full, 0, 1, axis=0)
    #     # print("St_full =", St_full)
    #     return transition_matrix @ St_full + \
    #            multivariate_normal.rvs(self.err_mean, self.err_cov)
    #
    # def transition2(self, t):
    #     '''
    #     Generate transition after change point
    #     '''
    #     At1 = 2 * self.At - 1
    #     transition_matrix = np.array([[10 - 0.6 * At1, 0.4 - 0.3 * At1, -0.04, -0.1, 0.1 + 0.1 * At1],
    #                                   [11 + 0.4 * At1, 0.05, 0.4, 0, 0],
    #                                   [1.2 + 0.5 * At1, -0.02, 0.03 - 0.03 * At1, 0.8, 0]])
    #     St_full = np.insert(self.St, 3, self.Stm1, axis=0)
    #     St_full = np.insert(St_full, 0, 1, axis=0)
    #     # print("St_full =", St_full)
    #     return transition_matrix @ St_full + \
    #            multivariate_normal.rvs(self.err_mean, self.err_cov)
    #
    #
    # def simulate(self, seed=0, T0=0, T1=26, burnin = 0,
    #              optimal_policy_model=None, normalized = [0.0, 1.0]):
    #     '''
    #     simulate states, rewards, and action data
    #     :param mean0: mean vector for the initial state S_0
    #     :param cov0: covariance matrix for the initial state S_0
    #     :param seed: numpy random seed
    #     :param S0: initial states of N subjects
    #     :param A0: initial actions of N subjects
    #     :param optimal_policy_model: input actions and simulate state and rewards following the input actions. Can be used for
    #     Monte Carlo policy evaluation
    #     :return: a list [ states, rewards, actions]
    #     '''
    #     # if T1 > self.T:
    #     #     print("error: terminal time T1 of the simulated trajectory should be no more than the total number of time points T")
    #     #     return 0
    #
    #     # set seed
    #     np.random.seed(seed)
    #     States = np.zeros([self.N, T1-T0 + 2, 4])  # S[i][t]
    #     # Rewards = np.zeros([self.N, T1-T0])  # R[i,t]
    #     Actions = np.zeros([self.N, T1-T0+1])  # Actions[i,t]
    #
    #     if optimal_policy_model is None:
    #         for i in range(self.N):  # for each individual
    #
    #             ## initial
    #             # generate initial state S_i0
    #             self.St = np.ones(3)
    #             self.St[0] = np.random.normal(20, 3) # step count of week t
    #             self.St[1] = np.random.normal(20, 2) # sleep of week t
    #             self.St[2] = np.random.normal(7, 1) # mood score of week t
    #             if burnin == 0:
    #                 States[i, 0, :3] = self.St
    #             # generate policy
    #             self.At = np.random.binomial(1, 0.25)
    #             # record S_t-1
    #             St_minus_1 = copy(self.St)
    #             self.Stm1 = copy(St_minus_1[0])
    #
    #             ## for the subsequent time points
    #             t_recorded = 0
    #             for t in range(-burnin, self.change_pt-T0):  # each time point t
    #                 if t >= 0:
    #                     States[i, t_recorded, :3] = self.St
    #                     Actions[i, t_recorded] = self.At
    #                     # print("t_recorded = ", t_recorded, "ONE")
    #                     # print("St = ", self.St)
    #                     t_recorded += 1
    #                 if t >= 1:
    #                     self.Stm1 = copy(St_minus_1[0])
    #                     St_minus_1 = copy(self.St)
    #                 # generate S_i,t+1
    #                 self.St = self.transition1(t)
    #                 # generate policy
    #                 self.At = np.random.binomial(1, 0.25)
    #                 # self.At = 1
    #             # print("burned")
    #             # self.St = copy(States[i, 1, :3])
    #             for t in range(0, T1 - self.change_pt+2):  # each time point t
    #                 if t >= 0:
    #                     if t_recorded <= T1-T0:
    #                         Actions[i, t_recorded] = self.At
    #                     States[i, t_recorded, :3] = self.St
    #                     # print("t_recorded = ", t_recorded, "TWO")
    #                     # print("St = ", self.St)
    #                     t_recorded += 1
    #                 self.Stm1 = copy(St_minus_1[0])
    #                 St_minus_1 = copy(self.St)
    #                 # generate S_i,t+1
    #                 self.St = self.transition2(t)
    #                 # generate policy
    #                 self.At = np.random.binomial(1, 0.25)
    #
    #
    #     #%% if optimal policy is input
    #     else:
    #         myState = np.zeros([1, 2, 4])
    #         for i in range(self.N):  # for each individual
    #             ## initial t=0
    #             # generate initial state S_i0
    #             self.St = np.ones(3)
    #             self.St[0] = np.random.normal(20, 3) # step count of week t
    #             self.St[1] = np.random.normal(20, 2) # sleep of week t
    #             self.St[2] = np.random.normal(7, 1) # mood score of week t
    #             if burnin == 0:
    #                 States[i, 0, :3] = self.St
    #             # record S_t-1
    #             St_minus_1 = copy(self.St)
    #             self.Stm1 = copy(St_minus_1[0])
    #             myState[0, 0, :3] = self.St
    #             myState[0, 0, 3] = self.Stm1
    #             myState[0, 0, :] -= normalized[0]
    #             myState[0, 0, :] /= normalized[1]
    #             # generate policy
    #             self.At = optimal_policy_model.predict(myState).opt_action[0]
    #             # self.At = 1
    #
    #             ## for the subsequent time points
    #             t_recorded = 0
    #             if T0 < self.change_pt:
    #
    #                 ## for the subsequent time points
    #                 t_recorded = 0
    #                 # if T0 <= self.change_pt:
    #                 for t in range(-burnin, self.change_pt - T0):  # each time point t
    #                     # print("xxx")
    #                     if t == 0:
    #                         # print('t =', t_recorded, ", first")
    #                         States[i, t_recorded, :3] = self.St
    #                     elif t >= 1:
    #                         Actions[i, t_recorded] = self.At
    #                         States[i, t_recorded + 1, :3] = self.St
    #                         self.Stm1 = copy(States[i, t_recorded, 0])
    #                         # print('t =', t_recorded+1, ", first")
    #                         t_recorded += 1
    #                     # self.Stm1 = copy(self.St[0])
    #                     # generate S_i,t+1
    #                     self.St = self.transition1(t)
    #                     # generate action according to policy
    #                     myState[0, 0, :3] = self.St
    #                     myState[0, 0, 3] = self.Stm1
    #                     myState[0, 0, :] -= normalized[0]
    #                     myState[0, 0, :] /= normalized[1]
    #                     self.At = optimal_policy_model.predict(myState).opt_action[0]
    #
    #             # after change point
    #             for t in range(0, T1 - self.change_pt+2):  # each time point t
    #                 if t >= 0:
    #                     if t_recorded <= T1 - T0:
    #                         Actions[i, t_recorded] = self.At
    #                     States[i, t_recorded, :3] = self.St
    #                     # self.Stm1 = copy(States[i, t_recorded, 0])
    #                     # print('t =', t_recorded+1, ", second")
    #                     t_recorded += 1
    #                 self.Stm1 = copy(St_minus_1[0])
    #                 St_minus_1 = copy(self.St)
    #                 # print("self.At", self.At)
    #                 # generate S_i,t+1
    #                 self.St = self.transition2(t)
    #                 # generate policy
    #                 myState[0, 0, :3] = self.St
    #                 myState[0, 0, 3] = copy(St_minus_1[0])
    #                 myState[0, 0, :] -= normalized[0]
    #                 myState[0, 0, :] /= normalized[1]
    #                 self.At = optimal_policy_model.predict(myState).opt_action[0]
    #                 # self.At = 1
    #                 # print(myState)
    #                 # print(self.At)
    #             States[i, t_recorded, :3] = self.St
    #
    #     # add weekly step of week t-1 to S4
    #     States[:,1:,3] = States[:,:(T1-T0+1),0]
    #     States = np.delete(States, 0, axis=1)
    #     # make rewards
    #     Rewards = copy(States[:,:(T1-T0),0])
    #     # Rewards = copy(States[:,1:,0])
    #     # convert Actions to integers
    #     Actions = np.delete(Actions, 0, axis=1)
    #     Actions = Actions.astype(int)
    #     return States, Rewards, Actions
