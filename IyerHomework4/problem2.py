import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas import DataFrame 
from sklearn import datasets 
from sklearn.mixture import GaussianMixture
import sys

log = open("problem2.log", "w")
sys.stdout = log
iris = datasets.load_iris() # load the iris dataset 
X = iris.data[:, :4] # select first two columns  
d = pd.DataFrame(X) # turn it into a dataframe 
gmm = GaussianMixture(n_components = 3)
gmm.fit(d) 
print('Max iterations for convergence: ' + str(gmm.n_iter_))
print('Mean: ' + str(gmm.means_))
print('Covariance: ' + str(gmm.covariances_))
print('Weights: ' + str(gmm.weights_))
log.close()