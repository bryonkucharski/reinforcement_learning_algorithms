import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import sys
from collections import defaultdict
import itertools
import time

class ActorCritic:
    def __init__(self,weight_init,num_actions,env, gamma, alpha,beta,lmbda, use_func_approx,max_steps, epsilon, epsilon_decay, epsilon_min,action_selection_method = "e-greedy", temperature = None,func_approx_type = 'fourier', order=None, num_state_dimensions = None, num_states = None):
        self.weight_init = weight_init
        self.order = order

        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.lmbda = lmbda
        self.use_func_approx = use_func_approx
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.num_states = num_states
        self.num_state_dimensions = num_state_dimensions
        self.max_steps = max_steps
        self.func_approx_type = func_approx_type
        self.order = order

        self.weight_init = weight_init
        self.action_selection_method = action_selection_method
        self.temperature = temperature
        self.epsilon = self.epsilon_start


        if use_func_approx:
            self.order_list = _get_order_array(self.order, self.num_state_dimensions, start=0)

        self.reset()


    def reset(self):
        if self.use_func_approx:
            self.theta = self.weight_init*np.random.randn(self.num_actions, len(self.order_list))
            self.w =     self.weight_init*np.random.randn(len(self.order_list))
            self.elgibility_v = np.zeros(len(self.order_list))
            self.elgibility_theta = np.zeros((self.num_actions, len(self.order_list)))
        else:
            self.theta = self.weight_init*np.random.randn(self.env.num_actions * self.env.num_states)
            self.elgibility_v = np.zeros(self.env.num_states)
            self.elgibility_theta = np.zeros(self.env.num_actions * self.env.num_states)
            self.value = np.zeros(self.env.num_states)

    def update_actor_elgibility(self,state, action):
        if self.use_func_approx:

            #self.elgibility_theta = self.gamma * self.lmbda * self.elgibility_theta
            row = self.action_value_approximate(state)
            probs = self.softmax(row)
            pi_s = probs

            b = np.zeros((self.num_actions, len(self.order_list)))
            for i in range(self.num_actions):
                if i == action:
                    b[i] =  (1-pi_s[i]) * self.phi(state).reshape(1, -1)
                else:
                    b[i] = (-pi_s[i]) * self.phi(state).reshape(1, -1)
           # b[np.arange(len(b)) != action] = -pi_s_a_theta[]* self.phi(state).reshape(1, -1)
            self.elgibility_theta = self.gamma * self.lmbda * self.elgibility_theta + b

        else:

            action_weights = np.array_split(self.theta, self.num_states)[state]
            pi_s = self.softmax(action_weights)
            #print(np.sum(pi_s_a_theta))
            #assert np.sum(pi_s_a_theta) == 1.0

            #decay all traces
            for i in range(len(self.elgibility_theta)):
                self.elgibility_theta[i] = self.gamma*self.lmbda*self.elgibility_theta[i]

            #calculate dln(pi)
            #for the every action in vector for current state
            for i in range(self.num_actions):
                #add 1-pi to the action you took
                if i == action:
                    # the indexing here will get me to the start of the current action vector for current state. add i to index the current action
                    self.elgibility_theta[(self.num_actions*state) + i] += (1 - pi_s[i])
                #add -pi to actions you didnt take
                else:
                    self.elgibility_theta[(self.num_actions * state) + i] += pi_s[i]

    def update_critic_elgbility(self, state, action):
        if self.use_func_approx:
            self.elgibility_v = self.gamma * self.lmbda * self.elgibility_v + self.phi(state).reshape(1,-1)
        else:
            for i in range(len(self.elgibility_v)):
                self.elgibility_v[i] = self.gamma * self.lmbda * self.elgibility_v[i]
            # update current state by one
            self.elgibility_v[state] += 1



    def get_value_TD_error(self, r, state, state_prime,done):
        if self.use_func_approx:
            if done:
                pred = 0
            else:
                pred = self.value_approximate(state_prime)

            return r + self.gamma*pred - self.value_approximate(state)
        else:
            return (r + self.gamma*self.value[state_prime]) - self.value[state]

    def update_actor(self, td_error):
        self.theta += self.beta*td_error*self.elgibility_theta

    def update_critic(self,state,action, td_error):
        if self.use_func_approx:
            self.w = self.w + self.alpha*(td_error) * self.elgibility_v
        else:
            self.value = self.value + self.alpha*(td_error)*self.elgibility_v

    def phi(self,state):
        return fourier_basis(state, self.order_list)

    def value_approximate(self,state):
        return np.dot(self.w, self.phi(state))

    def action_value_approximate(self,state, action = None):
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

    def select_action(self,state):
        if self.use_func_approx:
            row = self.action_value_approximate(state)
        else:
            action_weights = np.array_split(self.theta, self.num_states)[state]  # split theta into chunks, return the actions that correspond to current s
            # subtract max for numerical stability
            row = action_weights - action_weights.max()
        #print(row)
        probs = self.softmax(row)
        #print(probs)
        return int(np.random.choice(self.num_actions, 1, p=probs))

    def softmax(self, row):
        probs = (np.exp((1 / self.epsilon) * row) / np.sum(np.exp((1 / self.epsilon) * row)))
        probs = probs.reshape(-1)
        return probs

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
        rewards = []
        steps = 0
        while True:
            a = self.select_action(s)
            s_, r, done = self.env.step(a)
            if self.use_func_approx:
                # s = self.env.normalize_state(s)
                s_ = self.env.normalize_state(s_)
            rewards.append(r)


            self.update_critic_elgbility(s,a)
            td_error = self.get_value_TD_error(r,s, s_, done)
            self.update_critic(s,a,td_error)

            self.update_actor_elgibility(s, a)
            self.update_actor(td_error)


            if steps >= self.max_steps:
                done = True

            #print(s,a,r,s_, done, td_error)
            if done:
                return rewards
            else:
                s = s_

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                steps += 1
