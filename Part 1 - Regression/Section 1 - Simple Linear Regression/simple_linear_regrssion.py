#Simple Linear Regression

#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read dataset
dataset = pd.read_csv('Salary_Data.csv');
X = dataset.iloc[:,0].values
y = dataset.iloc[:,-1].values

#processing missing data
'''from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy='mean')
imp_mean.fit(X[:,1:3])
X[:,1:3] = imp_mean.transform(X[:,1:3])'''

#processing label encode
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#label encoder
'''labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])'''

#Onhot Encoder
'''onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''

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

#visualising the test set result


