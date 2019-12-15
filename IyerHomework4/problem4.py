import numpy as np, pandas as pd 
import itertools, scipy.stats, seaborn as sns, matplotlib.colors as colors
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm
from matplotlib import pyplot as plt

sns.set()
iris = sns.load_dataset("iris") #loads the iris dataset
iris = iris.rename(index = str, columns = {'sepal_length':'sepal_length','sepal_width':'sepal_width', 'petal_length':'petal_length', 'petal_width':'petal_width'})
df = iris[["sepal_length", "sepal_width",'species']]

#Using first two features for three class classification
X_data = df.iloc[:,0:2]
y_labels = df.iloc[:,2].replace({'setosa':0,'versicolor':1,'virginica':2}).copy()

#Gaussian naive bayes classifier
model = GaussianNB(priors = None)
model.fit(X_data,y_labels) #fit data and y labels to classifier

X = np.linspace(4, 8, 100)
Y = np.linspace(1.5, 5, 100)
X, Y = np.meshgrid(X, Y)   

g = sns.FacetGrid(iris, hue="species", height=10, palette = 'colorblind').map(plt.scatter, "sepal_length", "sepal_width",).add_legend()
plot = g.ax

dim = np.array(  [model.predict( [[xx,yy]])[0] for xx, yy in zip(np.ravel(X), np.ravel(Y)) ] )
Z = dim.reshape(X.shape)

plot.contourf( X, Y, Z, 2, alpha = .1, colors = ('blue','green','red'))
plot.contour( X, Y, Z, 2, alpha = 1, colors = ('blue','green','red'))
plot.set_xlabel('Sepal Length')
plot.set_ylabel('Sepal Width')
plot.set_title('Gaussian Naive Bayes 3 class classifier')
plt.savefig('Problem4.jpg')