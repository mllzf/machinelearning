# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:37:47 2020

@author: DELL
"""

import pandas as pd
from numpy import *
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("C:\\Users\\DELL\\Desktop\\mydata.csv")
X_train,X_test,Y_train,Y_test = train_test_split(df.loc[:,['tobacco','ldl','age']],df.loc[:,['chd']],train_size=.80)
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)
print(classifier.coef_) 