import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read dataset from file csv
eur_usd = pd.read_csv("eur_usd.csv")
usd_index = pd.read_csv("dollar_index.csv")

eur_usd.iloc[:,0] = pd.to_datetime(eur_usd.iloc[:,0])
usd_index.iloc[:,0] = pd.to_datetime(usd_index.iloc[:,0])

eur_usd = eur_usd[eur_usd.Volume != 0]
usd_index = usd_index[usd_index.Volume != 0]
#merge two data
dataset = pd.DataFrame({'TIME':eur_usd['Local time'],'EUR_USD':eur_usd['Close'],'USD_INDEX':usd_index['Close']})

#splitting 
y = dataset.iloc[:,1:2].values
X = dataset.iloc[:,-1].values

#processing missing data
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy="mean")
X = imp_mean.fit_transform(X.reshape(-1,1)).flatten()

#plitting dataset into trainning set and test set
y = y.flatten()
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#processing Simple Linear Regression
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))

#predicting
y_predicting = linear_regression.predict(X_test.reshape(-1,1)).flatten()

#visualizing 
plt.scatter(X_train, y_train, c='r')
plt.plot(X_train,linear_regression.predict(X_train.reshape(-1,1)).flatten(),c="b")
plt.title('USD Index AND USD/EUR')
plt.xlabel('USD INDEX')
plt.ylabel('EUR/USD')
plt.show()

plt.scatter(X_test, y_test, c='r')
plt.plot(X_train,linear_regression.predict(X_train.reshape(-1,1)).flatten(),c="b")
plt.title('USD Index AND USD/EUR')
plt.xlabel('USD INDEX')
plt.ylabel('EUR/USD')
plt.show()