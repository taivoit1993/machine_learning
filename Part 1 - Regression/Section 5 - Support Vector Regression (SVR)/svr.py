import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,-1].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1,1)).flatten()
y = sc_Y.fit_transform(y.reshape(-1,1)).flatten()

#Fitting SVR
from sklearn.svm import SVR
regression = SVR(kernel="rbf")
regression.fit(X.reshape(-1,1),y.reshape(-1,1))

#Prediction result
y_pred = sc_Y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))

#Visualising the SVR results
plt.scatter(X, y , c='r')
plt.plot(X, regression.predict(X.reshape(-1,1)).flatten(),c='b')
plt.title("Truth or Bluff (SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()