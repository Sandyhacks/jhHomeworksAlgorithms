import numpy as np
import pandas as pd
import sys
import itertools, scipy.stats, seaborn as sns, matplotlib.colors as colors
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn import preprocessing, datasets, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from scipy.stats import norm
from scipy import stats
from neupy.algorithms import PNN

if __name__ == "__main__":
  print('Starting program. See plots and logs for results in a few seconds.')
  # log = open("logs.log", "w")
  # sys.stdout = log

  try:
    #Reading in dataset
    print('Reading dataset\n\n')
    dataset = pd.read_csv('iris_data_for_cleansing.csv')

    # Step 1 - Cleaning data
    print('Step 1: Data cleansing')
    column_names = dataset.columns
    for i in column_names:
      print('{} is unique: {}'.format(i, not(dataset[i].is_unique))) #Check if every value is unique
    for column in column_names:
        dataset[column] = dataset[column].fillna(dataset[column].mean())
        assert(dataset[column] != str ).any() #Check if every value is not a string
        assert(dataset[column] >= 0 ).all() #Check if every value is greater than zero
    dataset = dataset.dropna() #remove rows with NAN values
    print('Data cleansing complete\n\n')

    # Step 2 - Generate two sets of features from the original 4 features to end up with a total of 8 features
    # using first two features to generate two new features
    minimum = dataset.min() #get min of dataset
    maximum = dataset.max() #get max of dataset
    mean  = dataset.mean() #get mean of dataset
    mean_list = [mean[0], mean[1]] #list of means
    std = StandardScaler().fit_transform(dataset.iloc[:,0:2].values) #fit transform
    cov = np.cov(std.T) #covariance matrix
    x, y = np.random.multivariate_normal(mean_list, cov, len(dataset.index)).T #generating random numbers
    df = pd.DataFrame(list(zip(x, y)), columns =['generated 1', 'generated 2']) #creating dataframe
    scaler_x = preprocessing.MinMaxScaler(feature_range=(minimum[0], maximum[0])) #scaling
    scaler_y = preprocessing.MinMaxScaler(feature_range=(minimum[1], maximum[1])) #scaling
    df[['generated 1']] = scaler_x.fit_transform(df[['generated 1']]) #creating final dataframe
    df[['generated 2']] = scaler_y.fit_transform(df[['generated 2']]) #creating final dataframe 
    print('Step 2: Two sets of new features (total 8 features)')
    print(df)
    print('\n\n')

    # Step 3- To remove any outliers we can use the Z-score function from scipy
    print('Step 3: Removing outliers using z score function from scipy library')
    z = np.abs(stats.zscore(dataset))
    threshold = 3
    print('Check dataframe for where z score criteria is not met')
    print(np.where(z > threshold))
    dataset = dataset[(z<3).all(axis=1)].reset_index() #removing outliers
    target = dataset['class']
    y = dataset['class']
    print('Created new dataframe after performing feature preprocessing\n\n')
    dataset = dataset.drop(columns=['index', 'class'])

    # Step 4 - To rank the top two features, the chi-square test and f test is used. Both methods seem to give the same result
    print('Step 4: Ranking top two features from the set of 6 features. This outlines two methods, 1. chi-square test and 2. f test. Both methods yield same result')
    x_train, x_cv, y_train, y_cv=train_test_split(dataset, target, test_size=0.2, stratify=target) #split test and train set
    features_chi = [] 
    features_f = []

    sel_chi2 = SelectKBest(chi2, k=2)    # select 2 features, can be modified
    X_train_chi2 = sel_chi2.fit_transform(x_train, y_train)
    rank_chi = sel_chi2.get_support()
    for rank in range(len(rank_chi)): 
      if rank_chi[rank] == True:
        colname = dataset.columns[rank]
        features_chi.append(colname)

    sel_f = SelectKBest(f_classif, k=2) # select 2 features, can be modified
    X_train_f = sel_f.fit_transform(x_train, y_train)
    rank_f = sel_f.get_support()
    for rank in range(len(rank_f)): 
      if rank_f[rank] == True:
        colname = dataset.columns[rank]
        features_f.append(colname)
    print('Top two features: ')
    print(features_f)
    print('\n\n')

    #Step 5: PCA
    print('Step 5: Reducing dimentionality to two features using PCA. See plot')
    scaler = preprocessing.StandardScaler()
    scaler.fit(dataset)
    X_scaled_array = scaler.transform(dataset)
    X_scaled = pd.DataFrame(X_scaled_array, columns = dataset.columns)

    ndimensions = 2
    seed = 0
    pca = PCA(n_components=ndimensions, random_state=seed)
    pca.fit(X_scaled)
    X_pca_array = pca.transform(X_scaled)
    X_pca = pd.DataFrame(X_pca_array, columns=['PC1','PC2']) 
    finalDf = pd.concat([X_pca, target], axis = 1)
    finalDf.rename(columns = {'class':'target'}, inplace = True) 
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [1, 2, 3]
    colors = ['r', 'g', 'b']
    for t, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == t
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1'], finalDf.loc[indicesToKeep, 'PC2'], c = color, s = 50)
    ax.legend(targets)
    ax.grid()
    df_plot = X_pca.copy()
    plt.savefig('PCA.png')
    print('\n\n')

    # Step 6: Expectation maximization
    print('Step 6a: Expectation maximization. See plot.')
    gmm = GaussianMixture(n_components=3) # Gaussian mixture model that uses Expectation maximization algorithm
    gmm.fit(X_scaled)
    y_cluster_gmm = gmm.predict(X_scaled) # prediction
    df_plot['ClusterGMM'] = y_cluster_gmm
    groupby = 'ClusterGMM'
    fig, ax = plt.subplots(figsize = (7,7))
    cmap = mpl.cm.get_cmap('prism')
    for i, cluster in df_plot.groupby(groupby):
        cluster.plot(ax = ax, kind = 'scatter',x = 'PC1', y = 'PC2',color = cmap(i/(3-1)),label = "%s %i" % (groupby, i),s=30) 
    ax.grid()
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_title("Iris Dataset")
    plt.savefig('EM.png')
    print('\n\n')

    # Step 6: LDA
    print('Step 6b: Linear Discriminant Analysis. See plot.')
    lda = LinearDiscriminantAnalysis(n_components=2) # applies LDA
    X = dataset.as_matrix()
    y = y.as_matrix()
    X_r2 = lda.fit(X, y).transform(X)
    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    target_names = [1, 2, 3]
    for color, i, target_name in zip(colors, [1, 2, 3], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of IRIS dataset')
    plt.savefig('LDA.png')
    print('\n\n')

    # Step 6: Probabilistic NN
    print('Step 6c: Probabilistic neural network using neupy library for iris classification.')
    print("Starting classification of iris dataset")
    skfold = StratifiedKFold(n_splits=10) # Provides train/test indices to split data in train/test sets
    data = X
    target = y
    for i, (train, test) in enumerate(skfold.split(data, target), start=1):
        x_train, x_test = data[train], data[test]
        y_train, y_test = target[train], target[test]

        pnn_network = PNN(std=0.1, verbose=False) # applies PNN
        pnn_network.train(x_train, y_train) # trains network
        result = pnn_network.predict(x_test) #prediction

        n_predicted_correctly = np.sum(result == y_test)
        n_test_samples = test.size

        print("Test #{:<2}: Guessed {} out of {}".format(i, n_predicted_correctly, n_test_samples))
        print('Accuracy\n')
        print(metrics.accuracy_score(y_test, result)) # prints classifcation accuracy report
        print('\n')
    print('\n\n')

    # Step 6: SVM
    print('Step 6d: Support Vector machine classification.')
    model=SVC()
    model.fit(x_train, y_train)
    pred=model.predict(x_cv) # prediction using Support vector machine method
    print('Classification report and analysis using support vector machine')
    print(classification_report(y_cv, pred)) # prints classification report
    print('Program finished')
    # log.close()
    exit()

  except AssertionError as a:
    print('Assertion error found')
    print(a)
  except Exception as e:
    print('General Exception found')
    print(e)