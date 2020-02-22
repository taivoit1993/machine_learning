import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data from file csv
dataset = pd.read_csv('Position_Salaries.csv');
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

#fitting desicion tree regression to dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state=0)
regression.fit(X,y)

#prediction result
y_pred = regression.predict(np.array([[6.5]]))

#visualising the desicing tree regression results
plt.scatter(X,y,c='r')
plt.plot(X,regression.predict(X),c='b')
plt.title('Truth off Bluff(Decision Tree Regression)')
plt.xlabel('Position Level');
plt.ylabel('Salaries')
plt.show()

#visualising the desicing tree regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y,c='r')
plt.plot(X_grid, regression.predict(X_grid), c='b')
plt.title('Truth off Bluff(Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

