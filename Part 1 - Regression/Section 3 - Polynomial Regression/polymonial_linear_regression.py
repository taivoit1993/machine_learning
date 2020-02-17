#import the librabries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read dataset from csv
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#splitting dataset into trainning set and test set
from sklearn.model_selection import train_test_split
X_train, y_train , X_test , y_test = train_test_split(X , y, test_size = 0.2,random_state = 0)

#fitting linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fitting Polymonial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualising the linear regression results
plt.scatter(X,y,c='r')
plt.plot(X, lin_reg.predict(X),c='b')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

#Visualising the Polymonial regression results
plt.scatter(X, y, c='r')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),c='b')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')