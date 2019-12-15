import numpy as np 
import pandas as pd 
import random
import matplotlib.pyplot as plt , sys
from pandas import DataFrame 
from sklearn import datasets 
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing

log = open("problem3.log", "w")
sys.stdout = log
mean =  [4.5, 2.2, 3.3]
cov = [[0.5, 0.1, 0.05], [0.1, 0.25, 0.1], [0.05, 0.1, 0.4]]
minimum = [3.5, 1.7, 2.5]
maximum = [5.5, 2.7, 4.1]

x, y,z = np.random.multivariate_normal(mean, cov, 300).T
df = pd.DataFrame(list(zip(x, y, z)), columns =['0', '1', '2'])
scaler_x = preprocessing.MinMaxScaler(feature_range=(minimum[0], maximum[0]))
scaler_y = preprocessing.MinMaxScaler(feature_range=(minimum[1], maximum[1]))
scaler_z = preprocessing.MinMaxScaler(feature_range=(minimum[2], maximum[2]))
df[['0']] = scaler_x.fit_transform(df[['0']])
df[['1']] = scaler_y.fit_transform(df[['1']])
df[['2']] = scaler_z.fit_transform(df[['2']])
gmm = GaussianMixture(n_components = 3)
gmm.fit(df) 
print('Max iterations for convergence: ' + str(gmm.n_iter_))
print('Mean: ' + str(gmm.means_))
print('Covariance: ' + str(gmm.covariances_))
print('Weights: ' + str(gmm.weights_))
log.close()