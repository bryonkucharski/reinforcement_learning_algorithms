import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import sys
from collections import defaultdict
import itertools
import time

class TDLearning:
    def __init__(self,env,gamma, alpha, use_func_approx,func_approx_type = 'Fourier', order=None,num_state_dimensions = None):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.use_func_approx = use_func_approx
        if not self.use_func_approx:
            #create an array to hold values for every state
            self.value =  np.zeros(env.num_states)
        else:
            if func_approx_type == 'Fourier':
                self.order = order
                self.w = np.zeros((self.order + 1) ** num_state_dimensions)

    def get_TD_error(self, state,r, state_prime):
        if self.use_func_approx:
            return r + self.gamma*self.value_approximate(state_prime) - self.value_approximate(state)
        else:
            return (r + self.gamma*self.value[state_prime]) - self.value[state]
    
    def update(self, state,td_error):
        if self.use_func_approx:
            self.w = self.w + self.alpha*(td_error) * self.phi(state)
        else:
            self.value[state] = self.value[state] + self.alpha*(td_error)
           
    def phi(self, state):
        '''
        Convert state to order-th Fourier basis 
        '''
        phi = []
        j = 0
        for i in itertools.product(np.arange(self.order + 1),repeat=len(state)):
            i = np.array(i)
            
            scalar = np.dot(i,state)
            val = np.cos(np.pi*scalar)
            phi.append(val)
            j += 1
        
      
        return np.array(phi)
    
    def value_approximate(self,state):
        '''
        Computes V(s) using Function Approximation
        '''
        return np.dot(self.w.T,self.phi(state))

    def train(self, num_steps):
        '''
        let it run for num_steps episodes to update weights/values before evaluating 
        '''

        for i in range(num_steps):
            s = self.env.reset()
            while(True):
                a = self.env.get_random_action() 
            
               
                if self.use_func_approx:
                    s_, r,done = self.env.step(a)
                    state = self.env.normalize_state(s)
                    state_prime = self.env.normalize_state(s_)
                else:
                    s_, r,done = self.env.step(s,a)
                    state = self.env.convert_tuple_state_to_single_state(s)
                    state_prime = self.env.convert_tuple_state_to_single_state(s_)
                
                error = self.get_TD_error(state,r,state_prime)
                self.update(state,error)
                #print(state,state_prime,error,self.value[state],self.value[state_prime])
                if done:
                    break
                else:
                    s = s_
                
    def episode(self):
        '''
        Run after evaluation. Collect squred td errrors for entire episode. Return the average td error for this episode.
        '''
        s = self.env.reset()
        errors = []
        while(True):

            a = self.env.get_random_action() 
            
            if self.use_func_approx:
                s_, r,done = self.env.step(a)
                state = self.env.normalize_state(s)
                state_prime = self.env.normalize_state(s_)
            else:
                s_, r,done = self.env.step(s,a)
                state = self.env.convert_tuple_state_to_single_state(s)
                state_prime = self.env.convert_tuple_state_to_single_state(s_)
            
            error = self.get_TD_error(state,r,state_prime)

            sqrd_error = error**2
       
            errors.append(sqrd_error)

            if done:
                return np.average(errors)
            else:
                s = s_

class Q_Learning:
    def __init__(self,weight_init,num_actions, env, gamma, alpha, use_func_approx,max_steps, epsilon, epsilon_decay, epsilon_min,action_selection_method = "e-greedy", temperature = None,func_approx_type = 'fourier', order=None, num_state_dimensions = None):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.use_func_approx = use_func_approx
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.num_state_dimensions = num_state_dimensions
        self.max_steps = max_steps
        self.func_approx_type = func_approx_type
        self.order = order

        self.weight_init = weight_init
        self.action_selection_method = action_selection_method
        self.temperature = temperature
        self.epsilon = self.epsilon_start

        if self.use_func_approx:
            if self.func_approx_type == 'RBF':
                self.order_list = _get_order_array(self.order,self.num_state_dimensions,start = 0)
            else:
                self.order_list = _get_order_array(self.order,self.num_state_dimensions,start = 0)
        self.reset()

    def reset(self):
        #self.epsilon = self.epsilon_start

        if not self.use_func_approx:
            self.q_table = defaultdict(lambda: [0.0 for x in range(self.num_actions) ])
        else:
            #self.w = np.zeros( (self.num_actions, (self.order + 1) ** self.num_state_dimensions) )
            self.w = self.weight_init*np.random.randn(self.num_actions, len(self.order_list))


    def phi(self, state):
        if self.func_approx_type == 'fourier':
            return fourier_basis(state,self.order_list)
        elif self.func_approx_type == 'polynomial':
            return polynomial_basis(state,self.order_list)
        elif self.func_approx_type == 'RBF':
            return radial_basis_function(state,self.order_list,self.order,self.temperature)

    def action_value_approximate(self,state,action = None):
        '''
        Computes Q(s,a) using Function Approximation
        Returns one approximation for each action if action is None
        If there is an action, returns the Q(s,a) for that action
        '''
        Q_s = np.dot(self.w,self.phi(state))
        assert Q_s.shape == (self.num_actions,1)
        if action == None:
            return Q_s
        else:
            return Q_s[action]

    def get_TD_error(self, state, action, r, state_prime,done):
        if not self.use_func_approx:
            return r + self.gamma*(max(self.q_table[state_prime]) - self.q_table[state][action])
        else:
            if done:
                pred =  0
            else:
                pred = max(self.action_value_approximate(state_prime))
            return r + self.gamma*(pred) - self.action_value_approximate(state, action)
    
    def update(self, state, action, td_error):
        if self.use_func_approx:
            a =  self.alpha*(td_error)* self.phi(state).reshape(1,-1)
            b = np.zeros((self.num_actions,len(self.order_list)))
            b[action] = a
            self.w += b
            
        else:
            self.q_table[state][action] += self.alpha*(td_error)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def select_action(self, state):
        if self.action_selection_method == "e-greedy":
            rnd = np.random.rand()
            if rnd <= self.epsilon:
                action = np.random.choice(range(self.num_actions))
            else:
                if not self.use_func_approx: #do a row lookup

                    row = list(self.q_table[state])

                else: #approximate from phi*weights
                    row = self.action_value_approximate(state)

                action = np.random.choice(np.flatnonzero(row == np.max(row)))#select max from row, break ties by randomly selecting one of the actions
                #print(state, row, action)

        elif self.action_selection_method == "softmax":

            row = self.action_value_approximate(state)
            probs = (np.exp((1/self.epsilon)* row) / np.sum(np.exp((1/self.epsilon) * row)))
            probs = probs.reshape(-1)
            #print(probs)
            action = int(np.random.choice(self.num_actions, 1, p=probs ))
        return action
        

    def get_return(self,trajectory):
        """
        Calcualte discounted future rewards base on the trajectory of an entire episode
        """
        r = 0.0
        for i in range(len(trajectory)):
            r += self.gamma**i * trajectory[i]
       
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
                #s = self.env.normalize_state(s)
                s_ = self.env.normalize_state(s_)

            rewards.append(r)
            td_error = self.get_TD_error(s,a,r,s_,done)
            #print(self.action_value_approximate(s))
            self.update(s,a,td_error)
            #print(self.action_value_approximate(s))
            #print()

            if steps >= self.max_steps:
                done = True

            #print(s,a,r,s_, done,td_error)
            if done:
                return rewards
            else:
                s = s_
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                steps += 1

            self.env.render()

            

    def print_q_table(self):
        '''
        prints all values currently in the q table
        ''' 

        for item in sorted(self.q_table.items()):
            print(str(item))

class Sarsa:
    def __init__(self,weight_init,num_actions, env, gamma, alpha, use_func_approx,max_steps, epsilon, epsilon_decay, epsilon_min, action_selection_method = 'e-greedy',temperature = None, func_approx_type = 'fourier', order=None, num_state_dimensions = None):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
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
        self.reset()
        self.epsilon = self.epsilon_start
        self.action_selection_method = action_selection_method
        self.temperature = temperature

        if self.use_func_approx:
            self.order_list =  _get_order_array(self.order,self.num_state_dimensions)#used when calculating

    def reset(self):
        #self.epsilon = self.epsilon_start

        if not self.use_func_approx:
            self.q_table = defaultdict(lambda: [0.0 for x in range(self.num_actions) ])
        else:
            #self.w = np.zeros( (self.num_actions, (self.order + 1) ** self.num_state_dimensions) )
            self.w = self.weight_init*np.random.randn(self.num_actions, (self.order + 1) ** self.num_state_dimensions)


    def phi(self, state):
        if self.func_approx_type == 'fourier':
            return fourier_basis(state, self.order_list)
        elif self.func_approx_type == 'polynomial':
            return polynomial_basis(state,self.order_list)
        elif self.func_approx_type == 'RBF':
            return radial_basis_function(state,self.order_list,self.order,self.temperature)

    def get_TD_error(self, state, action, r, state_prime, action_prime,done):
        if not self.use_func_approx:
            return r + self.gamma*(self.q_table[state_prime][action_prime] - self.q_table[state][action])
        else:
            if done:
                pred = 0
            else:
                #print(self.action_value_approximate(state, action), self.action_value_approximate(state_prime,action_prime))
                pred = self.action_value_approximate(state_prime,action_prime)

            return r + self.gamma*(pred) - self.action_value_approximate(state, action)
    

    
    def action_value_approximate(self,state,action = None):
        '''
        Computes Q(s,a) using Function Approximation
        Returns one approximation for each action if action is None
        If there is an action, returns the Q(s,a) for that action
        '''

        Q_s = np.dot(self.w,self.phi(state))
        assert Q_s.shape == (self.num_actions,1)
        if action == None:
            return Q_s
        else:
            return Q_s[action]
    
    def update(self, state, action, td_error):
        if self.use_func_approx:
            a =  self.alpha*(td_error)* self.phi(state).reshape(1,-1)
            b = np.zeros((self.num_actions,(self.order + 1) ** self.num_state_dimensions))
            b[action] = a
            self.w += b
        else:
            self.q_table[state][action] += self.alpha*(td_error)

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
    
    def get_return(self,trajectory):
        """
        Calcualte discounted future rewards base on the trajectory of an entire episode
        """
        r = 0.0
        for i in range(len(trajectory)):
            r += self.gamma**i * trajectory[i]
       
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
                #s = self.env.normalize_state(s)
                s_ = self.env.normalize_state(s_)

            rewards.append(r)
            a_ = self.select_action(s_)
            td_error = self.get_TD_error(s,a,r,s_,a_,done)

            #print(self.w[s])
            self.update(s,a,td_error)
            #print(self.w[s])

            if steps >= self.max_steps:
                done = True

            #print(s,a,r,s_, done, td_error)
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

class Q_LearningLambda:
    def __init__(self, weight_init, num_actions, env, gamma, alpha,lmbda, use_func_approx, max_steps, epsilon, epsilon_decay,
                 epsilon_min, action_selection_method="e-greedy", temperature=None, func_approx_type='fourier',
                 order=None, num_state_dimensions=None):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lmbda = lmbda

        self.use_func_approx = use_func_approx
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.num_state_dimensions = num_state_dimensions
        self.max_steps = max_steps
        self.func_approx_type = func_approx_type
        self.order = order

        self.weight_init = weight_init
        self.action_selection_method = action_selection_method
        self.temperature = temperature
        self.epsilon = self.epsilon_start

        if self.use_func_approx:
            self.order_list = _get_order_array(self.order, self.num_state_dimensions, start=0)

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

    def get_TD_error(self, state, action, r, state_prime, done):
        if not self.use_func_approx:
            return r + self.gamma * (max(self.q_table[state_prime]) - self.q_table[state][action])
        else:
            if done:
                pred = 0
            else:
                pred = max(self.action_value_approximate(state_prime))
            return r + self.gamma * (pred) - self.action_value_approximate(state, action)
    def update_elgibility(self,state,action):
        if self.use_func_approx:
            b = np.zeros((self.num_actions, len(self.order_list)))
            b[action] = self.phi(state).reshape(1,-1)
            self.elgibility = self.gamma*self.lmbda*self.elgibility  + b

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
            a = self.alpha * (td_error) * self.elgibility
            self.w += a

        else:
            self.q_table[state][action] += self.alpha * (td_error)*self.elgibility[state][action]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

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
        rewards = []
        steps = 0
        while True:
            a = self.select_action(s)
            s_, r, done = self.env.step(a)

            if self.use_func_approx:
                s_ = self.env.normalize_state(s_)

            rewards.append(r)

            self.update_elgibility(s,a)
            td_error = self.get_TD_error(s, a, r, s_, done)

            self.update(s, a, td_error)

            if steps >= self.max_steps:
                done = True

            # print(s,a,r,s_, done,td_error)
            if done:
                return rewards
            else:
                s = s_
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                steps += 1

            # self.env.render()

    def print_q_table(self):
        '''
        prints all values currently in the q table
        '''

        for item in sorted(self.q_table.items()):
            print(str(item))

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

def _get_order_array(order,number_of_states,start = 0):
    arr = []
    for i in itertools.product(np.arange(start,order + 1),repeat=(number_of_states)):
        arr.append(np.array(i))
    return np.array(arr)

def fourier_basis(state, order_list):
    '''
    Convert state to order-th Fourier basis 
    '''

    state_new = np.array(state).reshape(1,-1)
    scalars = np.einsum('ij, kj->ik', order_list, state_new) #do a row by row dot product with the state. i = length of order list, j = state dimensions, k = 1
    assert scalars.shape == (len(order_list),1)
    phi = np.cos(np.pi*scalars)
    return phi

def polynomial_basis(state, order_list):
    '''
    phi = []
    print(state.shape)
    for i in range(len(order_list)):
        c_i = order_list[i]
        scalar = np.prod(np.power(state,c_i))
        phi.append(scalar)
    return np.array(phi)
    '''
    state = np.array(state).reshape(1, -1)
    pows = np.power(state,order_list)
    phi = np.prod(pows,axis=1,keepdims=True)
    assert phi.shape == (len(order_list), 1)
    return phi

def radial_basis_function(state,order_list,order,sigma):

    state = np.array(state).reshape(1, -1)
    #c = order_list * (1/order)
    c = order_list
    subs = np.subtract(c,state)
    #sigma = 2/(order-1)
    norms_squared = np.power(np.linalg.norm(subs,axis=1,keepdims=True),2)

    a_k = np.exp(-norms_squared / (sigma * 2))*(1/np.sqrt(2*np.pi*sigma))
    phi = a_k
    #phi = a_k / np.sum(a_k)
    assert phi.shape == (len(order_list), 1)
    return phi

def cross_entropy_policy_search(env,num_iterations, K, elite_K, N, esp, filename, filename2, use_multiprocess = True):
    """
    Inputs:
        env - what enviornment to run policy search
        covariance_matrix - nxn covariance matrix
        K - population
        elite_K - elite population where elite_K  < K
        N - number of episodes per policy
        esp - stability parameter

    Outputs:
        best policy
        best covariance matrix
    """


    f = open(filename, "w")
    f2 = open(filename2, "w")
    policy_size = env.num_actions*env.num_states
    theta = 1e-0*np.random.randn(policy_size)
    cov = np.eye(policy_size)
    average_costs = []
    for i in range(num_iterations):
    
        all_returns = []
        results = []
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        #threads = [None] * K
        #results = [None] * K
        total_cost = 0.0
        for k in range(K):

            if use_multiprocess:
                p = mp.Process(target=worker, args=(N,theta,cov,env,return_dict,k))
                jobs.append(p)
                p.start()
            else:
                
                #print('Running index ' + str(k))
                theta_k = np.random.multivariate_normal(theta,cov)
                cost_k = env.simulate(N,action_selection = 'softmax', theta = theta_k, sigma = 1.0)
                total_cost += cost_k

                results.append((cost_k,theta_k))
                all_returns.append((i,k,cost_k))
                f.write(str(i) + ", " + str(k)  +", "+ str(cost_k) + "\n")
        
        if use_multiprocess:
            for job in jobs:
                job.join()
            
            results =  list(return_dict.items())  
            k = 0
            for cost_k,theta_k in results:
                total_cost += cost_k
                f.write(str(i) + ", 0, " + str(cost_k) + "\n")

        elite= sorted(results, key=lambda tup: tup[0], reverse=True)[:elite_K]
      
        total = 0.0
        diffs = 0.0
        total_elite_cost = 0.0
        for j in range(elite_K):
            total_elite_cost += elite[j][0]
            total += elite[j][1]
            diffs += np.dot((elite[j][1] - theta).reshape(-1,1), (elite[j][1] - theta).reshape(1,-1))
           

        theta = (1/elite_K) * total
        cov = (1/(elite_K + esp)) * ((esp * np.eye(policy_size)) + diffs)

        average_elite_costs = (total_elite_cost / elite_K)
        average_cost = total_cost / K
        f2.write( str(i) + ", " + str(average_cost) + ", " + str(average_elite_costs) + "\n" )
        print("Iteration: " + str(i) + " Average Cost: " + str(average_cost) + " Average Elite Cost: " + str(average_elite_costs))

        
    f.close()
    f2.close()
    return theta, cov, average_costs

def first_choice_hill_climbing(env, num_iterations, softmax_sigma,cov_sigma, N, verbose = False, filename = ''):

    """
    Inputs:
        env:            enviornment to run policy search
        num_iterations: how many iterations to run policy search for
        cov_sigma:      exploration parameter
        softmax_sigma:  scales how differences in theta changes the action probabilities
        N:              number of iterations to run evaulation for

    Outputs:
        best policy
    """
    f = open(filename, "w")
    policy_size = env.num_actions*env.num_states
    theta = 1e-0*np.random.randn(policy_size)
    cov = cov_sigma*np.eye(policy_size)
    cost = env.simulate(N,action_selection = 'softmax', theta = theta, sigma = softmax_sigma) #evaluate
    for i in range(num_iterations):
        #print(filename + str(i)
        np.random.seed(random.randint(0,123456789))
        theta_prime = np.random.multivariate_normal(theta,cov) #same a new set of theta
        cost_prime = env.simulate(N,action_selection = 'softmax', theta = theta_prime, sigma = softmax_sigma) #evaluate
        if verbose:
            print(filename + ": Iteraton: " + str(i) + " cost: " + str(cost) + " cost_prime: " + str(cost_prime))
        if cost_prime > cost:
            theta = theta_prime
            cost = cost_prime
        f.write(str(i) + ", " + str(cost) + ", " + str(cost_prime)+ "\n")
        
    print(filename + ': Best J: ' + str(cost))

    f.close()
        
    

    return theta

def worker(N,theta,cov,env,return_dict,index):
    '''worker function'''
    
    theta_k = np.random.multivariate_normal(theta,cov)
    cost_k = env.simulate(N,action_selection = 'softmax', theta = theta_k, sigma = 1.0)

    #for multithreading
    #return_dict[index] = (cost_k, theta_k)

    #multiprocessing
    return_dict[cost_k] = theta_k
    #f.write(str(i) + ", " + str(k)  +", "+ str(cost_k) + "\n")

    #print("Finished index " + str(index))

