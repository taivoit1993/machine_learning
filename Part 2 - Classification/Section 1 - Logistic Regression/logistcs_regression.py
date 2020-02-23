# import important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read dataset from file csv
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,2:4].values
y = dataset.iloc[:,-1].values

# spliting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# Featured Scaller
from sklearn.preprocessing import StandardScaler
X_scaller = StandardScaler()
X_train = X_scaller.fit_transform(X_train)
X_test = X_scaller.transform(X_test)

# fitting logistics Regression to Training Set
from sklearn.linear_model import LogisticRegression
classifyier = LogisticRegression(random_state=0)
classifyier = classifyier.fit(X_train, y_train)

# Prediction the result
y_pred = classifyier.predict(X_test)

# Making confusing Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#visualising the result
from matplotlib.colors import ListedColormap
X_set , y_set = X_train, y_train
X1 , X2 = np.meshgrid(np.arange( start = X_set[:,0].min() -1, stop = X_set[:,0].max()+1, step = 0.01),
                            np.arange(X_set[:,1].min() -1, X_set[:,1].max()+1,0.01))
plt.contourf(X1, X2, classifyier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()