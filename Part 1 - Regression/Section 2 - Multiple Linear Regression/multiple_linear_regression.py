#Multiple Linear Regression

#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read dataset and split variable 
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[: ,4].values

#Label Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])


onehotencoder_X = OneHotEncoder(categorical_features=[3])
X = onehotencoder_X.fit_transform(X).toarray()

#spliting the dataset into the Trainning set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

# Fitting Multiple Linear Regression to Trainning Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting The Test Set
y_pred = regressor.predict(X_test)



#Building the optimal modal using backward Elimination
X = X[:,1:]
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X,axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

#Remove model with p-value > 0.05
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(y,X_opt).fit()
regressor_OLS.summary()

#auto backwardElimination
def backWardElimination(x,y, sl):
    numVars = len(x[0])
    for i in range(0,numVars):
        regressor_OLS = sm.OLS(y,x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0,numVars - i):
                if regressor_OLS.pvalues[j].astype(float) == maxVar:
                    x = np.delete(x,j,1)
    regressor_OLS.summary()
    return x

X_opt = X[:,[0,1,2,3,4,5]]
X_Modeled = backWardElimination(X_opt, y, 0.05)
                    