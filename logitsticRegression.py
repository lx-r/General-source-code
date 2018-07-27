#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:54:18 2018
@author: rd
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math

def loadData():
    tmp=np.loadtxt("data.csv",dtype=np.str,delimiter=",")
    data=tmp[1:,:].astype(np.float)
    np.random.shuffle(data)
    train_data=data[:int(0.7*len(data)),:]
    test_data=data[int(0.7*len(data)):,:]
    train_X=train_data[:,:-1]/50-1.0 #feature normalization[-1,1]
    train_Y=train_data[:,-1]
    test_X=test_data[:,:-1]/50-1.0
    test_Y=test_data[:,-1]
    return train_X,train_Y,test_X,test_Y

#pos=np.where(train_Y==1.0)
#neg=np.where(train_Y==0.0)
#plt.scatter(train_X[pos,0],train_X[pos,1],marker='o', c='b')
#plt.scatter(train_X[neg,0],train_X[neg,1],marker='x', c='r')
#plt.xlabel('Chinese exam score')
#plt.ylabel('Math exam score')
#plt.legend(['Not Admitted', 'Admitted'])
#plt.show()
    
#The sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(h,Y):
    return (-Y*np.log(h)-(1-Y)*np.log(1-h)).mean()

def predict(X,theta,threshold):
    bias=np.ones((X.shape[0],1))
    X=np.concatenate((X,bias),axis=1)
    z=np.dot(X,theta)
    h=sigmoid(z)
    pred=(h>threshold).astype(float)
    return pred

def logisticRegression(X,Y,alpha,num_iters):
    model={}
    bias=np.ones((X.shape[0],1))
    X=np.concatenate((X,bias),axis=1)
    theta=np.ones(X.shape[1])
    for step in xrange(num_iters):
        z=np.dot(X,theta)
        h=sigmoid(z)
        grad=np.dot(X.T,(h-Y))/Y.size
        theta-=alpha*grad
        if step%1000==0:
            z=np.dot(X,theta)
            h=sigmoid(z)
            print "{} steps, loss is {}".format(step,loss(h,Y))
            print "accuracy is {}".format((predict(X[:,:-1],theta,0.5)==Y).mean()) 
    model={'theta':theta}
    return model
    
train_X,train_Y,test_X,test_Y=loadData()
model=logisticRegression(train_X,train_Y,alpha=0.01,num_iters=40000)
pre_Y=predict(train_X,model['theta'],0.5)
pos=np.where(train_Y==1.0)
neg=np.where(train_Y==0.0)
posp=np.where(pre_Y==1.0)
negp=np.where(pre_Y==0.0)
plt.scatter(train_X[pos,0],train_X[pos,1],marker='o', c='b')
plt.scatter(train_X[neg,0],train_X[neg,1],marker='x', c='r')
plt.scatter(train_X[posp,0],train_X[posp,1],marker='*', c='y')
plt.scatter(train_X[negp,0],train_X[negp,1],marker='+', c='g')
plt.xlabel('Chinese exam score')
plt.ylabel('Math exam score')
plt.legend(['Not Admitted', 'Admitted'])
plt.show()
print "The test accuracy is {}".format((predict(test_X,model['theta'],0.5)==test_Y).mean())   