#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
from sklearn import svm
df=pd.read_csv('data1.txt',sep=' ',header=None,dtype=str,na_filter=False)
trainlabels=df[1]
trainfeats=[]
i=0
while i<len(df):                   #处理每个特征值，比如将2:1202处理成1202
    strr=df.loc[i]
    j=2
    while j<25:
        str1=strr[j]
        if j>10:
            strr[j]=float(str1[3:])
        else: strr[j]=float(str1[2:])
        j+=1
    trainfeats.append(strr[2:])
    i+=1
trainfeats=np.array(trainfeats)
clf=svm.SVC()
clf.fit(trainfeats,trainlabels)                 #训练得到model
tf=pd.read_csv('data2.txt',sep=' ',header=None,dtype=str,na_filter=False)
name=tf[0]
testfeats=[]
i=0
while i<len(tf):               #处理特征值数据
    strr=tf.loc[i]
    j=1
    while j<24:
        str1=strr[j]
        if j>9:
            strr[j]=float(str1[3:])
        else: strr[j]=float(str1[2:])
        j+=1
    testfeats.append(strr[1:])
    i+=1
testfeats=np.array(testfeats)
predicts=clf.predict(testfeats)       #由特征值预测出类别
for i in range(len(name)):           #把类别标签写入文件
    if predicts[i]=='+1':
        predicts[i]='1'
    row=name[i]+' '+predicts[i]+'\n'
    open_file=file('solutionBySVM.txt','a+')
    open_file.write(row)
open_file.close()
print "done!"
  
  
  
  
  
  
  
  
