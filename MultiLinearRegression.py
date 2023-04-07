# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:05:27 2023

@author: JESSICA
"""

#Multiple Linear Regression

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data=pd.read_csv('finalexam.csv')
x = data[['EXAM1','EXAM2','EXAM3']]
y = data.FINAL

#Training the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.25, random_state = 6)
lr = LinearRegression()
lr.fit(x_train, y_train)

#predictions
lr_pred = lr.predict(x_test)

#finding residuals
print("RMSE :", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R^2: ", r2_score(y_test, lr_pred))

#predictions for new data
new=lr.predict([[66,69,65]])
print(new)

import pickle
with open('model2.pkl','wb') as files:
    pickle.dump(lr,files)