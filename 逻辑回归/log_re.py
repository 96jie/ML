# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 07:51:45 2018

@author: 96jie
"""

import numpy as np

label = []
feature = np.zeros([32561,123])
f = open('data.txt')
line = f.readline()
a = 0
while line:
    data = []
    for i in line.split( ):
        data.append(i);
    for i in data[1:]:
        j = i.split(":")
        feature[a][int(j[0]) - 1] = int(j[1])
    if data[0] in '+1':
       label.append(1) 
    else:
       label.append(0)         
    line = f.readline()
    a += 1
f.close
n = len(label)
label = np.mat(label)


#构建训练集和测试集
label1 = label[:,20001:32561]
label = label[:,0:20000]
feature1 = feature[20001:32561]
feature = feature[0:20000]
one1 = np.ones(len(feature1))
one = np.ones(len(feature))
feature1 = np.insert(feature1, 0, values=one1, axis=1)
feature = np.insert(feature, 0, values=one, axis=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 

def Optimization(feature,label,theta,learning_rate):
    for i in range(iter):
        theta = Updata(feature,label,theta,learning_rate)
    return theta

def Updata(feature,label,theta,learning_rate):
    h = 0
    alpha = learning_rate
    h = sigmoid(np.dot(feature , theta))
    theta += alpha * np.dot((label.transpose()-h).transpose(),feature).transpose()
    return theta

def acc(theta,feature,label):
    la = sigmoid(np.dot(feature,theta))
    la = la.tolist()
    m = len(la)
    label2 = []
    label = label[0].tolist()
    num = 0
    for c in range(len(la)):
        d = float(la[c][0])
        if d < 0.5:
            label2.append(0)
        else:
            label2.append(1)
        if(int(label[0][c]) != int(label2[c])):
            num += 1
    acc = 1-(num/m)    
    return acc

learning_rate = 0.0001
theta = np.ones([124,1])
iter = 1000 
theta = Optimization(feature,label,theta,learning_rate)
a = acc(theta,feature,label)
b = acc(theta,feature1,label1)
print(a)
print(b)

    
