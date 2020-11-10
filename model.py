import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pickle 

dataset=pd.read_csv('hiring/hiring.csv')
dataset['experience'].fillna(0,inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)

X=dataset.iloc[:,:3]

print(X)

y=dataset.iloc[:,-1]

print(y)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(X,y)


pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

print(model.predict([[2,9,6]]))