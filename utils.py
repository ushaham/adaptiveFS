# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:29:51 2019

@author: urixs
"""

import numpy as np
import gzip
import struct
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_data (case):
        
    if case == 122: # 50 questions
        data_file = "./Data/small_data50.npy"
        X = np.load(data_file)
        n, d = X.shape
        y = np.load('./Data/labels.npy')
        # standardize features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X) * 2 - 1
        question_names = np.load('./Data/names_small50.npy')
        class_names = ['no', 'yes']
        print('loaded data,  {} rows, {} columns'.format(n, d))
        
    if case == 123: # 100 questions
        data_file = "./Data/small_data100.npy"
        X = np.load(data_file)
        n, d = X.shape
        y = np.load('./Data/labels.npy')
        # standardize features
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X) * 2 - 1
        question_names = np.load('./Data/names_small100.npy')
        class_names = ['no', 'yes']
        print('loaded data,  {} rows, {} columns'.format(n, d))
        
        
    return X, y, question_names, class_names, scaler

def load_mnist(case=1):
    
    if os.path.exists('./mnist/X_test.npy'):
        X_test = np.load('./mnist/X_test.npy')
    else:
        X_test = read_idx('./mnist/t10k-images-idx3-ubyte.gz')
        X_test = X_test.reshape(-1, 28 * 28)
        np.save('./mnist/X_test.npy', X_test)
    if os.path.exists('./mnist/X_train.npy'):
        X_train = np.load('./mnist/X_train.npy')
    else:
        X_train = read_idx('./mnist/train-images-idx3-ubyte.gz')
        X_train = X_train.reshape(-1, 28 * 28)
        np.save('./mnist/X_train.npy', X_train)
    if os.path.exists('./mnist/y_test.npy'):
        y_test = np.load('./mnist/y_test.npy')
    else:
        y_test = read_idx('./mnist/t10k-labels-idx1-ubyte.gz')
        np.save('./mnist/y_test.npy', y_test)
    if os.path.exists('./mnist/y_train.npy'):
        y_train = np.load('./mnist/y_train.npy')
    else:
        y_train = read_idx('./mnist/train-labels-idx1-ubyte.gz')
        np.save('./mnist/y_train.npy', y_train)
        
    if case == 1: # small version
       train_inds = y_train <=2
       test_inds = y_test <= 2
       X_train = X_train[train_inds]
       X_test = X_test[test_inds]
       y_train = y_train[train_inds]
       y_test = y_test[test_inds]
    
    return X_train / 127.5 - 1., X_test / 127.5 - 1, y_train, y_test

def load_mi_scores():
    '''
    if os.path.exists('./mnist/mi.npy'):
        print('Loading stored mutual information scores')
        return np.load('./mnist/mi.npy')
    else:
        return None
    '''
    X_train, X_test, y_train, y_test = load_mnist(case=2)    
    max_depth = 5
    
    # define a decision tree classifier
    clf = DecisionTreeClassifier(max_depth=max_depth)    

    # fit model
    clf = clf.fit(X_train, y_train)
    return clf.feature_importances_

def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        
def plot_mnist_digit(digit, 
                     guess, 
                     true_label, 
                     num_steps, 
                     save=True, 
                     fig_num=0,
                     save_dir='.',
                     actions=None):
    import matplotlib.pyplot as plt
    digit = digit.reshape(28, 28)
    fig, ax = plt.subplots()
    ax.set_title('true label: {}, guess: {}, num steps: {}'.format(true_label, guess, num_steps), fontsize=18)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    im = ax.imshow(digit, cmap='gray')
    if actions is not None:
        for i, a in enumerate(actions):
            if a != 784:
                row = a % 28
                col = int(a / 28)
                text = ax.text(row, col - 2, i + 1, ha="center", va="center", color="b", size=15)  
    plt.show()
    if save:
        fig.savefig(save_dir + '/im_' + str(fig_num) +'.png')
        
def scale_individual_value(val, ind, scaler):
    return (val - scaler.data_min_[ind]) / (scaler.data_max_[ind] - scaler.data_min_[ind]) * 2. - 1.
        