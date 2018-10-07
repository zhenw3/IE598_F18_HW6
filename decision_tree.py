# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 14:13:44 2018

@author: zhenw
"""

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score 
from sklearn import metrics 

#load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

r=range(1,11)
score=[]
score1=[]
for i in r:
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    dt=DecisionTreeClassifier(criterion='gini', 
                              max_depth=4)
    dt.fit(X_train,y_train)
    score.append(dt.score(X_train,y_train))
    score1.append(dt.score(X_test,y_test))

dt=DecisionTreeClassifier(criterion='gini', 
                              max_depth=4)
score2=cross_val_score(dt,X_train,y_train,cv=10)
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred))

print("My name is Zhen Wang")
print("My NetID is: zhenw3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
    

