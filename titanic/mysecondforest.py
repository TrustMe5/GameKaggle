#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import cross_validation
import csv
df=pd.read_csv('../train.csv',header=0)
df=df.drop(['Ticket','Name','Cabin','Embarked'],axis=1)
m=np.ma.masked_array(df['Age'],np.isnan(df['Age']))  #把所有的Age取出来，Age为空的显示 '--'
mean=np.mean(m).astype(int)    #求平均值
df['Age']=df['Age'].map(lambda x:mean if np.isnan(x) else x)
df['Sex']=df['Sex'].map({'female':1,'male':0}).astype(int)
x=df.values
y=df['Survived'].values
x=np.delete(x,1,axis=1) #删除x数组的第二（下标为1）列
x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.3,random_state=0)
dt=tree.DecisionTreeClassifier(max_depth=5)
dt.fit(x_train,y_train)
print dt.score(x_test,y_test)
test=pd.read_csv('../test.csv',header=0)
tf=test.drop(['Ticket','Name','Cabin','Embarked'],axis=1)
m=np.ma.masked_array(tf['Age'],np.isnan(tf['Age']))
mean=np.mean(m).astype(int)
tf['Age']=tf['Age'].map(lambda x: mean if np.isnan(x) else int(x))
tf['Sex']=tf['Sex'].map({'female':1,'male':0}).astype(int)
tf['Fare']=tf['Fare'].map(lambda x: 0 if np.isnan(x) else int(x)).astype(int)
predicts=dt.predict(tf)
ids=tf['PassengerId'].values
prediction_file=open('./dt_submission.csv','wb')
open_file_object=csv.writer(prediction_file)
open_file_object.writerow(['PassengerId','Survived'])
open_file_object.writerows(zip(ids,predicts))
prediction_file.close()
