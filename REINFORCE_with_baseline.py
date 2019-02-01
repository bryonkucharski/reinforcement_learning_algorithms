import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import sys
from collections import defaultdict
import itertools
import time
import utils

class REINFORCE:
    def __init__(self,weight_init,num_actions,env, gamma, alpha,beta,lmbda, use_func_approx,max_steps, epsilon, epsilon_decay, epsilon_min,baseline = True, order=None, num_state_dimensions = None, num_states = None):
        self.use_func_approx = use_func_approx
        self.order = order
        self.weight_init = weight_init
        self.num_actions = num_actions
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.lmbda = lmbda
        self.max_steps = max_steps
        self.baseline = baseline
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_state_dimensions  = num_state_dimensions
        self.num_states = num_states

        if self.use_func_approx:
            self.order_list = _get_order_array(self.order,self.num_state_dimensions)


    def reset(self):
        if self.use_func_approx:
            self.theta = self.weight_init*np.random.randn(self.num_actions, len(self.order_list))
            self.w =     self.weight_init*np.random.randn(len(self.order_list))
        else:
            self.theta = self.weight_init*np.random.randn(self.env.num_actions * self.env.num_states)
            self.value = np.zeros(self.env.num_states)


    def episode(self):
        s = self.env.reset()
        if self.use_func_approx:
            s = self.env.normalize_state(s)
        trajectory = []
        rewards = []
        steps = 0
        while True:
            a = self.select_action(s)
            s_, r, done = self.env.step(a)
            rewards.append(r)
            if self.use_func_approx:
                # s = self.env.normalize_state(s)
                s_ = self.env.normalize_state(s_)

            if steps >= self.max_steps:
                done = True

            trajectory.append((s, a, r, s_, done))
            #print(s,a,r,s_, done, steps)
            if done:
                return trajectory, rewards
            else:
                s = s_
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                steps += 1

    def G_t(self,t, trajectory):
        rewards = 0
        for k in range(0,len(trajectory) - t):
            s,a,r,s_,done = trajectory[t + k]
            rewards += (self.gamma**k) * r

        return rewards

    def phi(self,state):
        return fourier_basis(state, self.order_list)

    def value_approximate(self,state):
        return np.dot(self.w, self.phi(state))

    def action_value_approximate(self, state, action=None):
        '''
        Computes Q(s,a) using Function Approximation
        Returns one approximation for each action if action is None
        If there is an action, returns the Q(s,a) for that action
        '''

        Q_s = np.dot(self.theta, self.phi(state))
        assert Q_s.shape == (self.num_actions, 1)
        if action == None:
            return Q_s
        else:
            return Q_s[action]

    def update_J(self,t,episode):

        s,a,r,s_, done = episode[t]

        if self.use_func_approx:
            row = self.action_value_approximate(s)
            probs = self.softmax(row)
            pi_s = probs

            b = np.zeros((self.num_actions, len(self.order_list)))
            for i in range(self.num_actions):
                if i == a:
                    b[i] = (1 - pi_s[i]) * self.phi(s).reshape(1, -1)
                else:
                    b[i] = (-pi_s[i]) * self.phi(s).reshape(1, -1)
            self.del_J += self.gamma**t * (self.G_t(t,episode) - self.value_approximate(s)) * b
        else:
            action_weights = np.array_split(self.theta, self.num_states)[s]
            pi_s = self.softmax(action_weights)

            if self.baseline:
                val = self.value[s]

            for i in range(self.num_actions):
                #add 1-pi to the action you took
                if i == a:
                    self.del_J[(self.num_actions * s) + i] +=  (self.G_t(t,episode) - val) * (1-pi_s[i])
                else:
                    self.del_J[(self.num_actions * s) + i] += (self.G_t(t, episode) - val) * (- pi_s[i])

    def update_w(self,TD_error):
        if self.use_func_approx:
            self.w = self.w + self.alpha*(TD_error) * self.elgibility
        else:
            self.value = self.value + self.alpha*(TD_error)*self.elgibility

    def update_elgibility(self,state):
        if self.use_func_approx:
            self.elgibility = self.gamma * self.lmbda * self.elgibility + self.phi(state).reshape(1,-1)
        else:
            self.elgilibty = self.elgibility * self.gamma*self.lmbda
            self.elgibility[state] += 1

    def update_theta(self):

        self.theta += self.beta*self.del_J

    def get_value_TD_error(self,r,s,s_,done):
        if self.use_func_approx:
            if done:
                pred = 0
            else:
                pred = self.value_approximate(s_)

            return r + self.gamma * pred - self.value_approximate(s)
        else:
            return (r + self.gamma * self.value[s_]) - self.value[s]

    def select_action(self,s):
        if self.use_func_approx:
            row = self.action_value_approximate(s)
        else:
            action_weights = np.array_split(self.theta, self.num_states)[s]  # split theta into chunks, return the actions that correspond to current s
            # subtract max for numerical stability
            row = action_weights - action_weights.max()
        #print(row)
        probs = self.softmax(row)
        #print(row, probs)
        return int(np.random.choice(self.num_actions, 1, p=probs))

    def softmax(self, row):
        probs = (np.exp((1 / self.epsilon) * row) / np.sum(np.exp((1 / self.epsilon) * row)))
        probs = probs.reshape(-1)
        return probs

    def train(self,num_episodes, debug=True):
        total_returns = []
        self.reset()
        for i in range(num_episodes):
            episode, rewards = self.episode()
            total_returns.append(self.G_t(0,episode))
            if debug:
                print(i , self.G_t(0,episode))

            if self.use_func_approx:
                self.del_J = self.weight_init * np.random.randn(self.num_actions, len(self.order_list))
                self.elgibility = np.zeros(len(self.order_list))
            else:
                self.del_J = np.zeros(self.env.num_actions * self.env.num_states)
                self.elgibility = np.zeros(self.env.num_states)

            for t in range(len(episode)):

                s, a, r, s_, done = episode[t]

                self.update_J(t,episode)

                if self.baseline:
                    self.update_elgibility(s)
                    TD_error = self.get_value_TD_error(r,s,s_,done)
                    self.update_w(TD_error)

            self.update_theta()#gradient ascent on J estimate
        return total_returns
