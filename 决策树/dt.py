# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 10:37:10 2018

@author: 96jie
"""

import numpy as np
from math import log
label = []
feature = np.zeros([32561,124])
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
       feature[a][123] = 1      
    else:
       feature[a][123] = 0      
    line = f.readline()
    a += 1
f.close

#构建训练集和测试集
feature1 = feature[5001:32561]
feature = feature[0:5000]

#每个特征中存在的取值
def classnums(data):
    cla = []
    for i in data:
       if i not in cla:
            cla.append(i)
    return cla

#计算经验熵
def calentropy(data):
    names = locals()
    n = len(data)
    p = []
    H = 0
    cla = classnums(data)
    for i in cla:
        names['class%s' %i] = data.count(i)
        p.append(data.count(i)/n)
    for i in p:
        H -= i*log(i,2)
    return H

#计算经验条件熵
def calentropy2(data1,data2):
    h = 0
    n = len(data1)
    names = locals()
    dic = {}
    cla = classnums(data1)
    cla1 = classnums(data2)
    for i in cla1:
        names['class%s' %i] = []    
    for i in range(n):
        a = data1[i]
        b = data2[i]
        dic[b] = names['class%s' %b] 
        names['class%s' %b].append(a)
    for i in dic:
        H = calentropy(dic[i])
        m = len(dic[i])
        H = (m / n) * H
        h = h + H
    return h

#切割数据集   
def spiltdata(data,axis,value):
    newdata = []
    for i in data:     
        if i[axis] == value:
            onedata = np.delete(i,axis,axis=0)
            newdata.append(onedata)
    newdata = np.array(newdata,dtype = float)
    return newdata

#计算信息增益或者信息增益比
def chooseBestFeature(feature,c):
    feature_nums = len(feature[0]) - 1
    label = feature[:,-1]
    label = label.tolist()
    feature = feature[:,0:feature_nums]
    h = calentropy(label)
    hbest = 0
    global feature_idx
    for i in range(feature_nums):
        a = feature[:,i]
        H = calentropy2(label,a)
        if c == 'id3':
            h1 = h - H
        if c == 'c4.5':
            h1 = (h-H)/h
        if hbest < h1:
            hbest = h1
            feature_idx = i
    #print(feature_idx)
    return feature_idx,hbest

#占比最大的类作为该结点的类标记
def maxkey(feature):
    cla = classnums(feature)
    max = 0
    for i in cla:
        feature1 = feature.tolist()
        a = feature1.count(i)
        if a > max:
            max = a
            maxkey = i
    return maxkey

#生成树
def createtree(feature,thre,c):
    if len(feature[0]) == 1:
        return maxkey(feature)
    label = feature[:,-1]
    cla = classnums(label)
    if len(cla) == 1:
        return cla[0]
    bestfeature,hbest = chooseBestFeature(feature,c)
    #print(hbest)
    if hbest < thre:
        return maxkey(feature[:,bestfeature])
    tree = {bestfeature:{}}
    bestfeatureclass = classnums(feature[:,bestfeature])
    for i in bestfeatureclass:
        tree[bestfeature][i] = createtree(spiltdata(feature,bestfeature,i),thre,c)
    return tree

#分类
def classify(data,tree):
    a = list(tree.keys())[0]
    lasta = tree[a]
    key = data[a]
    vof = lasta[key]
    if type(vof) == dict:
        classlabel = classify(data,vof)
    else:
        classlabel = vof     
    return classlabel

#计算正确率
def acc(feature,a):
    n = len(feature)
    t = 0
    for i in range(n):
        data = feature[i,0:122]
        b = classify(data,a)
        c = feature[i,-1]
        if b == c:
            t += 1
    return t/n

Tree = createtree(feature,0.04,'id3')
print(Tree)
print(acc(feature,Tree))   
print(acc(feature1,Tree)) 
