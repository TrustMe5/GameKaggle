#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd
file1=open('solutionByKNN.txt','r')
file2=open('example_solution.txt','r')
content1=file1.readlines()
content2=file2.readlines()
print content1
k=0
for i in content1:
    if i in content2:
        k+=1
print k
