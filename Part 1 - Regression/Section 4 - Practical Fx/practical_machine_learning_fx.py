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
X = imp_mean.fit_transform(X.reshape(-1,1)).flater()
