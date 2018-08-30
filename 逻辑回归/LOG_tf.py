# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:27:55 2018

@author: asus
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

label = []
feature = np.zeros([32561,123])
f = open(r'D:\python_test\test\test\样本\train.txt')
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
#feature1 = np.insert(feature1, 0, values=one1, axis=1)
#feature = np.insert(feature, 0, values=one, axis=1)

#参数
numclass = 1
Iter = 6000
inputSize = 123

#指定好x和y的大小
X = tf.placeholder(tf.float32, shape = [None, inputSize])
y = tf.placeholder(tf.float32, shape = [None, numclass])
#参数初始化
W1 = tf.Variable(tf.ones([123, 1]))
B1 = tf.Variable(tf.constant(0.1), [numclass])

#损失函数
y_pred = 1 / (1 + tf.exp(-tf.matmul(X, W1) + B1))
loss = tf.reduce_mean(- y * tf.log(y_pred) - (1 - y) * tf.log(1 - y_pred))
#训练
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.05)
train = opt.minimize(loss)
y1 = tf.round(y_pred)
#计算准确率
correct_prediction = tf.equal(y1, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(Iter):
    batchInput = feature
    batchLabels = label.transpose()
    sess.run(train, feed_dict={X: batchInput, y: batchLabels})
    if i%1000 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={X: batchInput, y: batchLabels})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
        
#测试集 
testAccuracy = sess.run(accuracy, feed_dict={X: feature1, y: label1.transpose()})
print ("test accuracy %g"%(testAccuracy))