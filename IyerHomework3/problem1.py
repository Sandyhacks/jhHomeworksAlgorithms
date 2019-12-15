import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from scipy.stats import trim_mean, skew
from statsmodels.formula.api import ols
import sys

log = open("problem1.log", "w")
sys.stdout = log

dataset = load_iris()
x = dataset['data']
y = dataset['target']
col_names = dataset['feature_names']
p = 0.1 # 10%
print('Trimmed mean value')
for i, col_name in enumerate(col_names):
    print(col_name, trim_mean(x[:,i], p))
data=pd.DataFrame(dataset['data'],columns=['Petal length','Petal Width','Sepal Length','Sepal Width'])
print('\n')
print(data.describe())
print('\nSKEW')
print(data.skew(axis = 0, skipna = True))
print('\nKURTOSIS')
print(data.kurtosis(axis=None, skipna=None))
log.close()