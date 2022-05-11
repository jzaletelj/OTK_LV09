# -*- coding: utf-8 -*-
"""
Created on Mon May 09 14:02:16 2016

@author: matevzk
"""

import numpy as np
from math import sqrt
import networkx as nx
import matplotlib.pyplot as plt


"""
Markov chain - splošne funkcije za generiranje, analizo in simulacijo
"""

def runSimulation(startState, transitMatrix, steps):
    # Nastavimo začetno stanje
    state = startState
    T = transitMatrix
    data = []; #tu zabeležimo vsak met (za histogram)
    # Izvajamo korake po verigi
    for i in range(1, steps+1):
        state = next_state(T[state])

        data.append(state+1) #prištejemo 1 ker python stanja šteje od 0 do 5
        
    return data
    
def next_state(weights):
    choice = np.random.random() * sum(weights)
    for i, w in enumerate(weights):
        choice -= w
        if choice < 0:
            return i
    
def getStateStats(data, relative = False):
    b = set(data)
    count = np.zeros(len(b), float);
    for i in data:
        count[i-1] += 1
    

    if relative:
        count /= len(data)
        
    return count
    
"""

Izdelava matrike prehodov (naključno ali na podlagi realnih podatkov)

"""    
    
def buildMatrix(random = True, data = [], minV = 1, maxV = 7, size = 100):
    if random:
        data = np.random.randint(minV,maxV,size)
    b = set(data)
    
    matrix = np.zeros((len(b),len(b)), float)
    
    
    for i in range((len(data)-1)):
        matrix[(data[i]-1)][(data[i+1]-1)] +=  1
        
             
    for i in b:
        matrix[i-1] = matrix[i-1]/float(sum(matrix[i-1]))
        
    return matrix
    
    
"""

Hidden Markov model, funkcije za izdelavo 

"""

def create_hidden_sequence(pi,A,length):
    out=[None]*length
    out[0]=next_state(pi)
    for i in range(1,length):
        out[i]=next_state(A[out[i-1]])
    return out

def create_observation_sequence(hidden_sequence,B,num):
    x = np.zeros((num,1));
    length=len(hidden_sequence)
    out=[None]*length
    for i in range(length):
        out[i]=next_state(B[hidden_sequence[i]])
        x[i] = out[i]
    return out
    
def group(L):
    first = last = L[0]
    for n in L[1:]:
        if n - 1 == last: 
            last = n
        else: 
            yield first, last
            first = last = n
    yield first, last 

def create_tuple(x):
    return [(a,b-a+1) for (a,b) in x]
    
"""
Primerjava modelov

"""

def getRMSE(orig,build):
    RMSE = 0.0
    n = orig.shape[0]
    for i in range(n):
        for j in range(n):
            RMSE += (orig[i][j]-build[i][j])**2
    
    RMSE = sqrt(RMSE)
    return RMSE

"""
Izris stanj
"""
def plotStates(T,labels):
    
    G = nx.from_numpy_matrix(np.array(T), create_using=nx.DiGraph())
    plt.figure()
    pos = nx.circular_layout(G)
    nx.draw_circular(G,node_size=900)
    nx.draw_networkx_labels(G,pos,labels,font_size=16)
    #plt.show()