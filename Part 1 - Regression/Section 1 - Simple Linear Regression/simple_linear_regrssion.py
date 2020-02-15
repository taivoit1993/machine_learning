#Simple Linear Regression

#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read dataset
dataset = pd.read_csv('Salary_Data.csv');
X = dataset.iloc[:,0].values
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Trainning set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

# Fiting Simple Linear Regression to the Trainning Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
print(regressor.intercept_)
print(regressor.coef_)
#predicting the Test set results
y_pred = regressor.predict(X_test.reshape(-1,1)).flatten()

#visualising the train and test result
plt.scatter(X_train, y_train, c='r');
plt.plot(X_train, regressor.predict(X_train.reshape(-1,1)).flatten(),c='b')
plt.title('Salary and Experience (Train)')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.show();

plt.scatter(X_test, y_test, c='r');
plt.plot(X_train, regressor.predict(X_train.reshape(-1,1)).flatten(),c='b')
plt.title('Salary and Experience (Test)')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.show();

