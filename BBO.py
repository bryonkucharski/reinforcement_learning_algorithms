import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import sys
from collections import defaultdict
import itertools
import time
import utils

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
