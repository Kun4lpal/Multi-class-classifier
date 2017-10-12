##############################################################################################################################################
# AUTHOR: KUNAL PALIWAL
# EMAIL ID: kupaliwa@syr.edu
# COURSE: ARTIFICAL NEURAL NETWORKS
# Assignment 2
#
# This implementation is done without using a cost matrix.
# The purpose is to compute accuracy
##############################################################################################################################################
from __future__ import division
import math
import random
import string
import pickle
import datetime
import time

random.seed(0)

# Some helper functions 

def rand(a, b):
    return (b-a)*random.random() + a

def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def func_tan(x):
    return math.tanh(x)

def dfunc_tan(y):
    return 1.0 - y**2

# Nerual network modelling class
# ni : input nodes
# nh : nodes in hidden layer
# no : nodes in output layer

class NN:
    # constructor
    def __init__(self, ni, nh, no):
        
        self.ni = ni + 1
        self.nh = nh
        self.no = no
        
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)
        
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    # update weights
    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('ValueError！')
        
        for i in range(self.ni-1):            
            self.ai[i] = inputs[i]
        
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = func_tan(sum)
        
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = func_tan(sum)
        
        return self.ao[:]

    # backpropogate to update weights
    def backPropagate(self, targets, N, M):                
        output_deltas = [0.0] * self.no
        correct = (targets.index(1))
        actual = (self.ao).index(max(self.ao))
        for k in range(self.no):
            error = (targets[k]-self.ao[k])            
            output_deltas[k] = dfunc_tan(self.ao[k]) * error

        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dfunc_tan(self.ah[j]) * error
        
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print(N*change, M*self.co[j][k])
        
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change
        
        error = 0.0        
        error += (0.5*(targets[k]-self.ao[k])**2)
        return error

    # Test my model
    def test(self, patterns):
        count = 0        
        for p in patterns:            
            target = (p[1].index(1))            
            result = self.update(p[0])            
            index = result.index(max(result))            
            count += (target == index)
            
        accuracy = float(count/len(patterns)) 
        print('accuracy: %-.9f' % (accuracy))


    # Print weights
    def weights(self):        
        for i in range(self.ni):
            print(self.wi[i])
        print()        
        for j in range(self.nh):
            print(self.wo[j])

    # Train my model
    def train(self, patterns, iterations=2000, N=0.005, M=0.01):
        # N: (learning rate)
        # M: (momentum factor)
        for i in range(iterations):
            error = 0.0
            for p in patterns:                
                inputs = p[0]
                targets = p[1]
                self.update(inputs)                
                error = error + self.backPropagate(targets, N, M)
            if(i%200==0):
                print('Error at iteration',i,':',error)            

import numpy as np
import pandas as pd

def config():
    ts = time.time()
    print('\n\n------------------------------------------------------------------------- Start time: ',datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),'------------------------------------------------------------')
    data = []
    # read dataset
    raw = pd.read_csv('X_train.csv')
    raw_data = raw.values    
    raw_feature = raw_data[0:,20:80]
    # Assigning the output vectors
    for i in range(len(raw_feature)-1):
        ele = []
        ele.append(list(raw_feature[i]))
        if raw_data[i][561] == 1:            
            ele.append([1,0,0,0,0,0])
        elif raw_data[i][561] == 2:            
            ele.append([0,1,0,0,0,0])
        elif raw_data[i][561] == 3:            
            ele.append([0,0,1,0,0,0])
        elif raw_data[i][561] == 4:            
            ele.append([0,0,0,1,0,0])
        elif raw_data[i][561] == 5:            
            ele.append([0,0,0,0,1,0])
        elif raw_data[i][561] == 6:            
            ele.append([0,0,0,0,0,1])
        data.append(ele)

    # shuffle data points and train
    random.shuffle(data)    
    training = data[0:100]
    test = data[500:600]
    nn = NN(60,20,6)
    nn.train(training,iterations=2000)
    ts = time.time()
    print('\n\n------------------------------------------------------------------------- Training finished at: ',datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),'------------------------------------------------------------')
    # now test my network
    nn.test(test)

if __name__ == '__main__':
    config()

