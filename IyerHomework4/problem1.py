import numpy as np, sys
import pandas as pd 
import random
import matplotlib.pyplot as plt 
from pandas import DataFrame 
from sklearn import datasets 
from sklearn.mixture import GaussianMixture

lst = [1, 4, 1, 4] 
lst2 = [2, 2, 3, 3]
df = pd.DataFrame(list(zip(lst, lst2)), columns =['0', '1'])
log = open("problem1.log", "w")
sys.stdout = log

for iteration in range(1,50):
    gmm = GaussianMixture(n_components = 3, max_iter = iteration, init_params='kmeans')
    gmm.fit(df) 
    if (iteration < 6):
        print('-----------------------------------------------')
        print('\n\nIteration number: ' + str(iteration))
        print('Mean: ' + str(gmm.means_))
        print('Covariance: ' + str(gmm.covariances_))
        print('Weights: ' + str(gmm.weights_))

print('-----------------------------------------------')
print('\n\nModel actually converged in iteration: ' + str(gmm.n_iter_))
log.close()