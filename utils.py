import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import sys
from collections import defaultdict
import itertools
import time


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