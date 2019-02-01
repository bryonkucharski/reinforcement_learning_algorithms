import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import sys
from collections import defaultdict
import itertools
import time
import utils

class SarsaLambda:
    def __init__(self, weight_init, num_actions, env, gamma, alpha,lmbda, use_func_approx, max_steps, epsilon, epsilon_decay,
                 epsilon_min, action_selection_method='e-greedy', temperature=None, func_approx_type='fourier',
                 order=None, num_state_dimensions=None):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lmbda = lmbda
        self.use_func_approx = use_func_approx
        self.func_approx_type = func_approx_type
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.num_state_dimensions = num_state_dimensions
        self.max_steps = max_steps
        self.order = order
        self.weight_init = weight_init

        self.epsilon = self.epsilon_start
        self.action_selection_method = action_selection_method
        self.temperature = temperature

        if self.use_func_approx:
            self.order_list = _get_order_array(self.order, self.num_state_dimensions)  # used when calculating

        self.reset()

    def reset(self):
        # self.epsilon = self.epsilon_start

        if not self.use_func_approx:
            self.q_table = defaultdict(lambda: [0.0 for x in range(self.num_actions)])
            self.elgibility = defaultdict(lambda: [0.0 for x in range(self.num_actions)])
        else:
            # self.w = np.zeros( (self.num_actions, (self.order + 1) ** self.num_state_dimensions) )
            self.w = self.weight_init * np.random.randn(self.num_actions, len(self.order_list))
            self.elgibility = np.zeros( (self.num_actions, len(self.order_list) ) )

    def phi(self, state):
        if self.func_approx_type == 'fourier':
            return fourier_basis(state, self.order_list)
        elif self.func_approx_type == 'polynomial':
            return polynomial_basis(state, self.order_list)
        elif self.func_approx_type == 'RBF':
            return radial_basis_function(state, self.order_list, self.order, self.temperature)

    def get_TD_error(self, state, action, r, state_prime, action_prime, done):
        if not self.use_func_approx:
            return r + self.gamma * (self.q_table[state_prime][action_prime] - self.q_table[state][action])
        else:
            if done:
                pred = 0
            else:
                # print(self.action_value_approximate(state, action), self.action_value_approximate(state_prime,action_prime))
                pred = self.action_value_approximate(state_prime, action_prime)

            return r + self.gamma * (pred) - self.action_value_approximate(state, action)

    def action_value_approximate(self, state, action=None):
        '''
        Computes Q(s,a) using Function Approximation
        Returns one approximation for each action if action is None
        If there is an action, returns the Q(s,a) for that action
        '''

        Q_s = np.dot(self.w, self.phi(state))
        assert Q_s.shape == (self.num_actions, 1)
        if action == None:
            return Q_s
        else:
            return Q_s[action]

    def update_elgibility(self,state,action):
        if self.use_func_approx:
            b = np.zeros((self.num_actions, len(self.order_list)))
            b[action] = self.phi(state).reshape(1, -1)
            self.elgibility = self.gamma * self.lmbda * self.elgibility + b

        else:

            #update every value in table that is not the current state by gamma*lambda*e
            prev_val = self.elgibility[state][action]
            for k, v in self.elgibility.items():
                for i in range(len(v)):
                    self.elgibility[k][i] = self.gamma*self.lmbda*self.elgibility[k][i]

            #update current state by one
            self.elgibility[state][action] = prev_val + 1

    def update(self, state, action, td_error):
        if self.use_func_approx:
            #a = self.alpha * (td_error) * self.phi(state).reshape(1, -1)
            #b = np.zeros((self.num_actions, (self.order + 1) ** self.num_state_dimensions))
            #b[action] = a
            #self.w += b
            a = self.alpha * (td_error) * self.elgibility
            self.w += a
        else:
            self.q_table[state][action] += self.alpha * (td_error)*self.elgibility[state][action]

    def select_action(self, state):
        if self.action_selection_method == "e-greedy":
            rnd = np.random.rand()
            if rnd <= self.epsilon:
                action = np.random.choice(range(self.num_actions))
            else:
                if not self.use_func_approx:  # do a row lookup

                    row = list(self.q_table[state])

                else:  # approximate from phi*weights
                    row = self.action_value_approximate(state)

                action = np.random.choice(np.flatnonzero(
                    row == np.max(row)))  # select max from row, break ties by randomly selecting one of the actions
                # print(state, row, action)

        elif self.action_selection_method == "softmax":

            row = self.action_value_approximate(state)
            probs = (np.exp((1 / self.epsilon) * row) / np.sum(np.exp((1 / self.epsilon) * row)))
            probs = probs.reshape(-1)
            # print(probs)
            action = int(np.random.choice(self.num_actions, 1, p=probs))
        return action

    def get_return(self, trajectory):
        """
        Calcualte discounted future rewards base on the trajectory of an entire episode
        """
        r = 0.0
        for i in range(len(trajectory)):
            r += self.gamma ** i * trajectory[i]

        return r

    def episode(self):
        s = self.env.reset()
        if self.use_func_approx:
            s = self.env.normalize_state(s)
        a = self.select_action(s)

        rewards = []
        steps = 0
        while True:

            s_, r, done = self.env.step(a)
            if self.use_func_approx:
                # s = self.env.normalize_state(s)
                s_ = self.env.normalize_state(s_)

            rewards.append(r)
            self.update_elgibility(s,a)
            a_ = self.select_action(s_)
            td_error = self.get_TD_error(s, a, r, s_, a_, done)

            # print(self.w[s])
            self.update(s, a, td_error)
            # print(self.w[s])

            if steps >= self.max_steps:
                done = True

            # print(s,a,r,s_, done, td_error)
            if done:
                return rewards
            else:
                s = s_
                a = a_
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                steps += 1

    def print_q_table(self):
        '''
        prints all values currently in the q table
        '''

        for item in sorted(self.q_table.items()):
            print(str(item))
