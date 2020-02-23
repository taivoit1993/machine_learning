import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

# Fitting Random Forest Regression into the dataset
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators=300,random_state=0 )
regression.fit(X,y)

#Prediction the result
y_pred = regression.predict(np.array([[6.5]]))

#visualising the result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y,c='r')
plt.plot(X_grid, regression.predict(X_grid), c='b')
plt.title('Truth off Bluff(Random forest regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()


