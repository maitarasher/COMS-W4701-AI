#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 09:51:17 2022

@author: maitarasher
"""

import numpy as np
from typing import Callable, Tuple, Dict, Set, List, Optional, cast
import matplotlib.pyplot as plt

data = np.array([[1,-1,-1,0],
                [1,0,1,0],
                [0,0,-1,0],
                [0,-1,1,0],
                [1,1,1,1],
                [0,1,0,1],
                [0,0,1,1],
                [-1,0,1,1]])


def count_data(data: np.array) ->  Tuple[np.array, np.array]:
    y_counts = np.zeros(2)
    f_counts = np.zeros((2,3,3))
    for r in range(data.shape[0]):
        y = data[r][-1]
        y_counts[y] += 1
        for c in range(data.shape[1]-1):
            x = data[r,c]
            f_counts[y,c,x] += 1
    return y_counts, f_counts

def CPT(class_counts: np.array, feature_counts: np.array, alpha: float) ->  Tuple[np.array, np.array]:
    y_cpts = class_counts.copy()
    f_cpts = feature_counts.copy()
    for r in range(feature_counts.shape[0]):
        for c in range(feature_counts.shape[1]):
            N = feature_counts[r,c,:].sum()
            f_cpts[r,c,:] += alpha
            f_cpts[r,c,:] /= (N + alpha*feature_counts.shape[2])
    N = y_cpts.sum()
    y_cpts += alpha
    y_cpts /= ( N + alpha*class_counts.shape[0])
    return y_cpts, f_cpts

def disribtion(data: np.array, class_cpt: np.array, feature_cpts: np.array) -> np.array:
    probs = np.zeros((data.shape[0],2))
    for i in range(data.shape[0]):
        sample = data[i]
        x1 = sample[0]
        x2 = sample[1]
        x3 = sample[2]
        for y in range(2):
            probs[i,y] = class_cpt[y]*feature_cpts[y,0,x1]*feature_cpts[y,1,x2]*feature_cpts[y,2,x3]
        probs[i,:] /= probs[i,:].sum()
    return probs
            
def linear_perceptron(data: np.array, alpha: float, y_tie: int) -> List[np.array]:
    w = np.zeros(4)
    weights = []
    converged = False
    mistakes: int
    while not converged:
        mistakes = 0
        for i in range(data.shape[0]):
            weights.append(w)
            sample = data[i]
            v = np.concatenate((np.array([1]),sample[0:3]))
            hw = np.dot(w,v)
            """predict class"""
            if hw < 0:
                y = 0
            elif hw > 0:
                y = 1
            else:
                y = y_tie
            """check if predicted class is equal to true class"""
            if y != sample[3]:
                mistakes += 1
                w = np.add(w,alpha*(sample[3] - y)*v)
        if mistakes == 0:
            converged = True
    return weights

def sigmoid_value(data: np.array, w) -> np.array:
    values = np.zeros(8)
    for i in range(data.shape[0]):
        sample = data[i]
        v = np.concatenate((np.array([1]),sample[0:3]))
        hw = 1 / (1 + np.exp(-1*np.dot(w,v)))
        values[i] = hw
    return values
    
    
if __name__ == "__main__":
    """Question 2"""
    alpha = 1
    y, f = count_data(data)
    print("y:\n",y,"\nf:\n", f)
    y_cpt, f_cpt = CPT(y,f,alpha)
    print("\ny cpts:\n", y_cpt, "\nf cpts:\n", f_cpt)
    probs = disribtion(data,y_cpt,f_cpt)
    print(probs[:,0])
    x= np.arange(probs.shape[0])
    plt.bar(x -0.1,probs[:,0], width=0.5, label="y=0")
    plt.bar(x +0.1,probs[:,1], width=0.5, label="y=1")
    plt.xticks(x)
    plt.legend()
    plt.xlabel('sample #')
    plt.ylabel('probability')
    plt.show()
    """Question 4"""
    a = 1
    weights = linear_perceptron(data,a,1)
    print("\nweights:\n", weights)
    plt.plot(weights, label=["index 0","index 1","index 2","index 3"])
    plt.legend()
    plt.xlabel('# of iterations')
    plt.ylabel('weight values')
    plt.show()
    w = weights[-1]
    values = sigmoid_value(data,w)
    print("sigmoid hw values:\n", values)
    plt.bar(x, values, width=0.5)
    plt.xlabel('sample #')
    plt.ylabel('hw')
    plt.show()
    
            