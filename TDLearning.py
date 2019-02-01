import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import sys
from collections import defaultdict
import itertools
import time
import utils

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
