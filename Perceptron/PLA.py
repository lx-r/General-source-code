#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 17:59:47 2018

@author: lx
"""
from __future__ import division
import pandas as pd
import numpy as np
import random

def getDate(filename):
    df=pd.read_csv(filename,delim_whitespace=True,names=['x0','x1','x2','x3','y'])
    x=np.asarray(df)
    random.shuffle(x)
    x_train=x[0:300,:-1]
    x_test=x[300:400,:-1]    
    y_train=x[:300,-1]
    y_test=x[300:400,-1]
    return x_train,y_train,x_test,y_test
def sign(x):
    return -1 if x<0 else 1;
def naive_PLA(X,Y,w,b,alpha,max_steps):
    num=len(X)
    flag=1
    step=0
    for i in xrange(max_steps):
        flag=1
        for j in xrange(num):
            y_=w.dot(X[j])+b
            if sign(y_)!=Y[j]:
                print "The loss of %d step is %5.5f."%(i,y_*Y[j])
                flag=0
                w+=alpha*Y[j]*X[j]
                b+=alpha*Y[j]
                break
            else:
                continue
        if flag==1:
            step=i
            break
    return w,b,step
def getAccuracy(X,Y,w,b):
    y_=[]
    for i in range(len(Y)):
        y0=sign(X[i].dot(w)+b)
        y_.append(y0)
    y_=np.array(y_,dtype=float)
    correct=np.flatnonzero(y_-Y)
    num=len(Y)
    return y_,len(correct)/num
    
if __name__=='__main__':
    filename='train_data.txt'
    x_train,y_train,x_test,y_test=getDate(filename)
#    w=np.random.random((4))
#    b=random.random()
    w=np.zeros((4,))
    b=0
    alpha=0.00001
    max_step=100000
    w,b,step=naive_PLA(x_train,y_train,w,b,alpha,max_step)
    print "The actual training step is {}".format(step)
    y_,acc=getAccuracy(x_test,y_test,w,b)
    print y_[:15]
    print y_test[:15]
    print "The accuracy of PLA is %2.4f%%."%((1.0-acc)*100)
