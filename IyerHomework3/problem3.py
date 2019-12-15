import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt    

def setosa(df):
    print(df)
    # Getting sub class dataframes and covariance matrices for each class
    setosa_df = df.loc[df['class'] == 'Iris-setosa']
    setosa_df = setosa_df.reset_index(drop=True)
    setosa_X = setosa_df.iloc[:,0:4].values
    setosa_X_std = StandardScaler().fit_transform(setosa_X)

    #get covariance matrix
    covariance_matrix_setosa = np.cov(setosa_X_std.T)

    #get mean,min,max
    setosa_df_details = setosa_df.describe()

    #creating 100 random numbers between 0 and 1
    setosa_random = np.random.uniform(0,1,size=(100, 4))

    #multiply covariance matrix with random matrix
    setosa_multiplied = setosa_random.dot(covariance_matrix_setosa)

    #normalize dataset
    setosa_dataframe = pd.DataFrame(setosa_multiplied)
    setosa_dataframe.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
    normalized_df = setosa_dataframe.copy()
    for feature_name in setosa_dataframe.columns:
        max_value = setosa_dataframe[feature_name].max()
        min_value = setosa_dataframe[feature_name].min()
        original_df_min = setosa_df_details.loc['min'][feature_name]
        original_df_max = setosa_df_details.loc['max'][feature_name]
        normalized_df[feature_name] = ((setosa_dataframe[feature_name] - min_value) / (max_value - min_value))*(original_df_max-original_df_min) + original_df_min

    #mean shifting   
    normalized_df_details = normalized_df.describe()
    for feature_name in setosa_dataframe.columns:
        mean_original = setosa_df_details.loc['mean'][feature_name]
        mean_normalized = normalized_df_details.loc['mean'][feature_name]
        mu = mean_normalized-mean_original
        normalized_df[feature_name] = normalized_df[feature_name] - mu

    return setosa_df, normalized_df

def virginica(df):
    # Getting sub class dataframes and covariance matrices for each class
    virginica_df = df.loc[df['class'] == 'Iris-virginica']
    virginica_df = virginica_df.reset_index(drop=True)
    virginica_X = virginica_df.iloc[:,0:4].values
    virginica_X_std = StandardScaler().fit_transform(virginica_X)
    covariance_matrix_virginica = np.cov(virginica_X_std.T)
    virginica_df_details = virginica_df.describe()
    #creating 100 random numbers between 0 and 1
    virginica_random = np.random.uniform(0,1,size=(100, 4))
    #multiply covariance matrix with random matrix
    virginica_multiplied = virginica_random.dot(covariance_matrix_virginica)

    #normalize dataset
    virginica_dataframe = pd.DataFrame(virginica_multiplied)
    virginica_dataframe.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
    normalized_df = virginica_dataframe.copy()
    for feature_name in virginica_dataframe.columns:
        max_value = virginica_dataframe[feature_name].max()
        min_value = virginica_dataframe[feature_name].min()
        original_df_min = virginica_df_details.loc['min'][feature_name]
        original_df_max = virginica_df_details.loc['max'][feature_name]
        normalized_df[feature_name] = ((virginica_dataframe[feature_name] - min_value) / (max_value - min_value))*(original_df_max-original_df_min) + original_df_min

    #mean shifting   
    normalized_df_details = normalized_df.describe()
    for feature_name in virginica_dataframe.columns:
        mean_original = virginica_df_details.loc['mean'][feature_name]
        mean_normalized = normalized_df_details.loc['mean'][feature_name]
        mu = mean_normalized-mean_original
        normalized_df[feature_name] = normalized_df[feature_name] - mu

    return virginica_df, normalized_df

def versicolor(df):
    # Getting sub class dataframes and covariance matrices for each class
    versicolor_df = df.loc[df['class'] == 'Iris-versicolor']
    versicolor_df = versicolor_df.reset_index(drop=True)
    versicolor_X = versicolor_df.iloc[:,0:4].values
    versicolor_X_std = StandardScaler().fit_transform(versicolor_X)
    covariance_matrix_versicolor = np.cov(versicolor_X_std.T)
    versicolor_df_details = versicolor_df.describe()
    #creating 100 random numbers between 0 and 1
    versicolor_random = np.random.uniform(0,1,size=(100, 4))
    #multiply covariance matrix with random matrix
    versicolor_multiplied = versicolor_random.dot(covariance_matrix_versicolor)

    #normalize dataset
    versicolor_dataframe = pd.DataFrame(versicolor_multiplied)
    versicolor_dataframe.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
    normalized_df = versicolor_dataframe.copy()
    for feature_name in versicolor_dataframe.columns:
        max_value = versicolor_dataframe[feature_name].max()
        min_value = versicolor_dataframe[feature_name].min()
        original_df_min = versicolor_df_details.loc['min'][feature_name]
        original_df_max = versicolor_df_details.loc['max'][feature_name]
        normalized_df[feature_name] = ((versicolor_dataframe[feature_name] - min_value) / (max_value - min_value))*(original_df_max-original_df_min) + original_df_min

    #mean shifting   
    normalized_df_details = normalized_df.describe()
    for feature_name in versicolor_dataframe.columns:
        mean_original = versicolor_df_details.loc['mean'][feature_name]
        mean_normalized = normalized_df_details.loc['mean'][feature_name]
        mu = mean_normalized-mean_original
        normalized_df[feature_name] = normalized_df[feature_name] - mu

    return versicolor_df, normalized_df

def save_plot(original_df, normalized_df, class_name, x_axis, y_axis):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    title = str(class_name) + '_' + str(x_axis) + '&' + str(y_axis)
    ax1.scatter(original_df[x_axis], original_df[y_axis], s=10, c='b', marker="s", label='Old IRIS set')
    ax1.scatter(normalized_df[x_axis], normalized_df[y_axis], s=10, c='r', marker="o", label='Generated IRIS set')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel(x_axis, fontsize=12)
    plt.ylabel(y_axis, fontsize=12)
    plt.savefig('problem3/'+str(title) + '.jpg')

    
def plot(original_df, normalized_df, class_name):

    #plot for combination 1
    x_axis = 'sepal_len'
    y_axis = 'sepal_wid'
    save_plot(original_df, normalized_df, class_name, x_axis, y_axis)

    #plot for combination 2
    x_axis = 'sepal_len'
    y_axis = 'petal_len'
    save_plot(original_df, normalized_df, class_name, x_axis, y_axis)

    #plot for combination 3
    x_axis = 'sepal_len'
    y_axis = 'petal_wid'
    save_plot(original_df, normalized_df, class_name, x_axis, y_axis)
   
    #plot for combination 4
    x_axis = 'sepal_wid'
    y_axis = 'petal_len'
    save_plot(original_df, normalized_df, class_name, x_axis, y_axis)

    #plot for combination 5
    x_axis = 'sepal_wid'
    y_axis = 'petal_wid'
    save_plot(original_df, normalized_df, class_name, x_axis, y_axis)

    #plot for combination 6
    x_axis = 'petal_len'
    y_axis = 'petal_wid'
    save_plot(original_df, normalized_df, class_name, x_axis, y_axis)

if __name__ == "__main__":
    #read in dataset
    df = pd.read_csv(
        filepath_or_buffer='iris.data', 
        header=None, 
        sep=',')
    df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    df.dropna(how="all", inplace=True) # drops the empty line at file-end

    class_name = 'setosa'
    setosa_original_df, setosa_normalized_df = setosa(df)
    plot(setosa_original_df, setosa_normalized_df, class_name)


    class_name = 'virginica'
    virginica_original_df, virginica_normalized_df = virginica(df)
    plot(virginica_original_df, virginica_normalized_df, class_name)

    class_name = 'versicolor'
    versicolor_original_df, versicolor_normalized_df = versicolor(df)
    plot(versicolor_original_df, versicolor_normalized_df, class_name)
    print('Done')