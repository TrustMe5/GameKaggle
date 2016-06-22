#!/usr/bin/env python
# coding=utf-8
import pandas as pd
import csv
tf=pd.read_csv('../test.csv',header=0)
ntf=tf.iloc[:,[0,3]]
ntf['Gender']=ntf['Sex'].map({'female':1,'male':0}).astype(int)
ids=ntf['PassengerId'].values
predicts=ntf['Gender'].values
predictions_file=open('./gender_submission.csv','wb')
open_file_object=csv.writer(predictions_file)
open_file_object.writerow(['PassengerId','Survived'])
open_file_object.writerows(zip(ids,predicts))
predictions_file.close()
