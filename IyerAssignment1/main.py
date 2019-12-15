'''
This program is developed by 'Sandesh Iyer' and does the following
1. Implements an algorithm to read in the Iris dataset
2. Implements an algorithm to visually see two sets of features and the class they belong to
3. Implements the sorting algorithm developed in Homework Problem 5

For step 1, Wwe read the input datafile using scipy loadarff module. After reading the dataset, we load the datasets
into a pandas dataframe. 
For step 2, the program uses the seaborn and matplotlib libraries to generate before sort and after sort graphs which will be downloaded
to the output folders 
For step 3, the program uses the in-built python sort_values method with mergesort as assigned sort method to sort each feature and plots
are downloaded to outputs folder once again

'''
from scipy.io import arff
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt    
import seaborn
import time, sys
seaborn.set(style='ticks')

'''
This function loads the dataset from iris.arff using the python loadarff module
'''
def get_iris_dataset(filename):
    iris_dataset = arff.loadarff('input/'+filename)
    return iris_dataset

'''
This function loads the iris_dataset tuple into a pandas dataframe 
'''
def get_dataframe(iris_dataset):
    iris_dataframe = pd.DataFrame(iris_dataset[0])
    return iris_dataframe

'''
This function sorts the iris dataset for each feature using python inbuilt sort_values method. 
The method is assigned to use mergesort method since its the most stable algorithm in comparison
to the other options for sort_values i.e. quicksort and heapsort
'''
def sort_iris_dataframe(feature_dataframe, feature_name):
    sorted_dataframe = feature_dataframe.sort_values(by = feature_name, kind='mergesort')
    return sorted_dataframe

'''
This function saves the seaborn plots for each feature(sepallength, sepalwidth, petallength, petalwidth) 
to before and after sort output folders
'''
def save_plot(feature_dataframe, feature_name, sort):
    plot_dataframe = pd.DataFrame({'points': feature_dataframe.index.values, feature_name: feature_dataframe[feature_name], 'class': feature_dataframe['class']})
    if sort == 'True':
        feature_dataframe = feature_dataframe.reset_index()
        plot_dataframe = pd.DataFrame({'points': feature_dataframe.index.values, feature_name: feature_dataframe[feature_name], 'class': feature_dataframe['class']})
        fg = seaborn.FacetGrid(data=plot_dataframe, hue='class', aspect=1.61)
        fg.map(plt.scatter, 'points', feature_name).add_legend()
        plt.title(feature_name + ' after sorting (feature in cm)')
        plt.savefig('output/after_sort_visuals/' + feature_name + '_sorted.jpg')
        plt.close()
    else:
        plot_dataframe = pd.DataFrame({'points': feature_dataframe.index.values, feature_name: feature_dataframe[feature_name], 'class': feature_dataframe['class']})
        fg = seaborn.FacetGrid(data=plot_dataframe, hue='class', aspect=1.61)
        fg.map(plt.scatter, 'points', feature_name).add_legend()
        plt.title(feature_name + ' before sorting (feature in cm)')
        plt.savefig('output/before_sort_visuals/' + feature_name + '.jpg')
        plt.close()

'''
Program starts here
'''
if __name__ == '__main__':
    start_time = time.time()
    print('Logs will appear in log.log file once program completes. Once program finished please see log file to see execution steps')
    print('Executing...')
    log = open("logs.log", "w")
    sys.stdout = log

    print('Loading iris dataset form iris.arff file using loadarff module')
    iris_dataset = get_iris_dataset('iris.arff')

    print('Loading iris dataset into pandas dataframe')
    iris_dataframe = get_dataframe(iris_dataset)

    print('Plotting before and after sort visualizations for each feature of iris dataset')
    for feature in iris_dataframe.columns:
        if feature != 'class':
            feature_dataframe = iris_dataframe.filter([feature, 'class'])
            feature_name = feature_dataframe.columns.values[0]

            print('\n-----' + feature_name.upper() + '-----')
            print('Before sort visualization for ' + feature_name + ' feature')
            sort = 'False'
            save_plot(feature_dataframe, feature_name, sort)

            print('Sorting...')
            sorted_dataframe = sort_iris_dataframe(feature_dataframe, feature_name)

            print('After sort visualization for ' + feature_name + ' feature')
            sort = 'True'
            save_plot(sorted_dataframe, feature_name, sort)
        else: 
            print('\nPLEASE SEE OUTPUTS FOLDER FOR BEFORE AND AFTER SORT VISUALIZATIONS OF IRIS DATASET\n')
    
    print('Exiting...')
    end_time = time.time()
    total_execution_time = end_time - start_time
    print('Program finished in ' + str(total_execution_time) + ' seconds')
    log.close()
    exit()


