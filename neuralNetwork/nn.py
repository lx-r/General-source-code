#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 19:11:28 2018

@author: rd
"""
from __future__ import division
import numpy as np

def sig(_z):
    _y=1/(1+np.exp(-_z))
    return _y

def predict(model,X):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1 
    a1 = np.tanh(z1) 
#    a1=sig(z1)
    z2 = a1.dot(W2) + b2 
    exp_scores = np.exp(z2) 
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

def get_accuracy(model,X,Y):
    probs=predict(model,X)
    pre_Y=np.argmax(probs,axis=1)
    comp=pre_Y==Y
    return len(np.flatnonzero(comp))/Y.shape[0]
    
def get_loss(model,X,Y,reg_lambda):
    probs=predict(model,X)
    # Calculating the loss 
    corect_logprobs = -np.log(probs[range(X.shape[0]), Y]) 
    data_loss = np.sum(corect_logprobs) 
    # Add regulatization term to loss  
    data_loss += reg_lambda/2 * (np.sum(np.square(model['W1']))+ np.sum(np.square(model['W2'])))
    loss = 1./X.shape[0] * data_loss     
    return loss
    
def nn_model(X,Y,nn_hdim,nn_output_dim,steps,epsilon,reg_lambda):
    np.random.seed(0) 
    W1 = np.random.randn(X.shape[1], nn_hdim)  
    b1 = np.ones((1, nn_hdim)) 
    W2 = np.random.randn(nn_hdim, nn_output_dim)
    b2 = np.ones((1, nn_output_dim)) 
    
    model={}
    
    for i in xrange(steps):
        ###forward propagation
        Z1=np.dot(X,W1)+b1
        a1=np.tanh(Z1)
#        a1=sig(Z1)
        Z2=np.dot(a1,W2)+b2
        #softmax output
        exp_score=np.exp(Z2)
        prob = exp_score/np.sum(exp_score,axis=1,keepdims=1)
        
        #Backward Propagation
        delta3=prob
        delta3[range(X.shape[0]),Y]-=1
        dW2 = np.dot(a1.T,delta3)
        delta2=np.dot(delta3,W2.T)*(1-np.power(a1,2))
        dW1 = np.dot(X.T,delta2)
        
        #update the weight value
        dW2+=reg_lambda*W2
        dW1+=reg_lambda*W1
        
        W2+=-epsilon*dW2
        W1+=-epsilon*dW1
        
        if i%500==0:
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            print "The {} steps, Loss = {:2.5f}, Accaracy = {:2.5f}".format(i,
                       get_loss(model,X,Y,reg_lambda),
                       get_accuracy(model,X,Y))
        
    return model
        
def main():
    """
    The data is saved in a 57x197 numpy array with a random order,
    197=14*14+1,14 is the image size, 1 is the label.
    """
    datas=np.load('data178x197.npy')
    np.random.seed(14)
    np.random.shuffle(datas)
    sp=int(datas.shape[0]/3)
    train_X=datas[:sp,:-1]
    train_Y=datas[:sp,-1]
    test_X=datas[sp:,:-1]
    test_Y=datas[sp:,-1]
    
    reg_lambda=0.2
    epsilon=0.00001
    steps=100000
    nn_output_dim=2   
    nn_hdim=14
    model=nn_model(train_X,train_Y,nn_hdim,nn_output_dim,steps,epsilon,reg_lambda)
    print"The test accuracy is {:2.5f}".format(get_accuracy(model,test_X,test_Y))
if __name__=='__main__':
    main()




