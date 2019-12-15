import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mlxtend.data import iris_data
from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

def plot(X_lda, title, y):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6))
        for lab, col in zip((0, 1, 2), ('blue', 'red', 'green')):
            plt.scatter(X_lda[y == lab, 0], X_lda[y == lab, 1], label=lab, c=col)
        plt.xlabel('Linear Discriminant 1')
        plt.ylabel('Linear Discriminant 2')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.title(title)
        plt.savefig('problem2/'+str(title) + '.jpg')

def plot2(X_lda, title, y):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(8, 6))
        for lab, col in zip((0, 1),('blue', 'red')):
            plt.scatter(X_lda[y == lab, 0], X_lda[y == lab, 1], label=lab, c=col)
        plt.xlabel('Linear Discriminant 1')
        plt.ylabel('Linear Discriminant 2')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.title(title)
        plt.savefig('problem2/'+str(title) + '.jpg')

if __name__ == "__main__":
    X, y = iris_data()
    X = standardize(X)  
    lda = LinearDiscriminantAnalysis(n_discriminants=2)
    lda.fit(X, y, n_classes=3)
    X_lda = lda.transform(X)
    title = 'Three class'
    plot(X_lda, title, y)

    y1 = y[:50]
    y2 = y[50:100]
    y3 = y[-50:]

    setosa = X[:50]
    versicolor = X[50:100]
    virginica = X[-50:]
    combination_1 = np.concatenate([setosa,versicolor])
    combination_2 = np.concatenate([versicolor,virginica])
    combination_3 = np.concatenate([setosa,virginica])
    y1_combo = np.concatenate([y1, y2])

    title = 'Two_class_setosa&versicolor'
    X1 = standardize(combination_1)
    lda = LinearDiscriminantAnalysis(n_discriminants=2)
    lda.fit(X1, y1_combo, n_classes=2)
    X1_lda = lda.transform(X1)
    plot2(X1_lda, title, y1_combo)

    title = 'Two_class_versicolor&virginica'
    X2 = standardize(combination_2)
    lda2 = LinearDiscriminantAnalysis(n_discriminants=2)
    lda2.fit(X2, y1_combo, n_classes=2)
    X2_lda = lda.transform(X2)
    plot2(X2_lda, title, y1_combo)

    title = 'Two_class_setosa&virginica'
    X3 = standardize(combination_3)
    lda3 = LinearDiscriminantAnalysis(n_discriminants=2)
    lda3.fit(X3, y1_combo, n_classes=2)
    X3_lda = lda.transform(X3)
    plot2(X3_lda, title, y1_combo)
