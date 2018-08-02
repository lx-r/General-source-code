#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:46:44 2018

@author: rd
"""
from __future__ import division
import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """
    def __init__(self):
        pass
    """In kNearestNeighbor,training means store the training data"""
    def train(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)
    
    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                 dists[i][j]=np.sum(np.square(self.X_train[j,:] - X[i,:]))
        return dists
    def compute_distances_one_loops(self,X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i]=np.sum(np.square(self.X_train-X[i]),axis=1)
        return dists
    def compute_distances_no_loops(self,X):
        squa_sum_X=np.sum(np.square(X),axis=1).reshape(-1,1)
        squa_sum_Xtr=np.sum(np.square(self.X_train),axis=1)
        inner_prod=np.dot(X,self.X_train.T)
        dists = -2*inner_prod+squa_sum_X+squa_sum_Xtr
        return dists
    
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            pos=np.argsort(dists[i])[:k]
            closest_y = self.Y_train[pos]
            y_pred[i]=np.argmax(np.bincount(closest_y.astype(int)))
        return y_pred
"""
This dataset is part of MNIST dataset,but there is only 3 classes,
classes = {0:'0',1:'1',2:'2'},and images are compressed to 14*14 
pixels and stored in a matrix with the corresponding label, at the 
end the shape of the data matrix is 
num_of_images x 14*14(pixels)+1(lable)
"""
def load_data(split_ratio):
    tmp=np.load("data216x197.npy")
    data=tmp[:,:-1]
    label=tmp[:,-1]
    mean_data=np.mean(data,axis=0)
    train_data=data[int(split_ratio*data.shape[0]):]-mean_data
    train_label=label[int(split_ratio*data.shape[0]):]
    test_data=data[:int(split_ratio*data.shape[0])]-mean_data
    test_label=label[:int(split_ratio*data.shape[0])]
    return train_data,train_label,test_data,test_label
def main():
    train_data,train_label,test_data,test_label=load_data(0.4)
    knn=KNearestNeighbor()
    knn.train(train_data,train_label)
    Yte=knn.predict(test_data,k=2)
    print "The accuracy is {}".format(np.mean(Yte==test_label))
if __name__=="__main__":
    main()