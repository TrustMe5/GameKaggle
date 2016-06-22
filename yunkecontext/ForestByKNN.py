#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
from numpy import *
import operator

df = pd.read_csv('data1.txt', sep=' ', header=None, dtype=str, na_filter=False)        #读取并处理训练集
group=[]
labels=[]
i=0
while i<len(df):     #处理特征值数据，比如将2:1202处理成1202
    strr=df.loc[i]
    j=2
    while j<25:
        str1=strr[j]
        if j>10:
           strr[j]=float(str1[3:])
        else: strr[j]=float(str1[2:])
        j+=1
    group.append(strr[2:])
    labels.append(strr[1])
    i+=1
group=np.array(group)

def classify0(inX, dataSet, labels, k):    #计算距离并得出所属类别
    dataSetSize = dataSet.shape[0]
    diffMat = (tile(inX, (dataSetSize,1))) -(dataSet)
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


tf = pd.read_csv('data2.txt', sep=' ', header=None, dtype=str, na_filter=False)           #读取并处理测试集
name=tf[0]
group2=[]
i=0
while i<len(tf):
    person=tf.loc[i]
    j=1
    while j<24:
        feat=person[j]
        if j>9:
           person[j]=float(feat[3:])
        else: person[j]=float(feat[2:])
        j+=1
    group2.append(person[1:])
    i+=1
group2=np.array(group2)
for i in range(len(group2)):        #写入文件
    outputlabel=classify0(group2[i],group,labels,50)   #得到类别标签
    if outputlabel=='+1':
        outputlabel='1'
    open_file=file('solutionByKNN.txt','a+')
    row=name[i]+" "+outputlabel+'\n'
    open_file.write(row)
open_file.close()
print "done!"
