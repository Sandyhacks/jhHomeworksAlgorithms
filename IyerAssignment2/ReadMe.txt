This program is developed in python3 by 'Sandesh Iyer' and does the following
1. Data Cleansing (use the iris data for cleansing.csv)
2. Generates two sets of features from the original 4 features to end up with a total of 8 featres
3. Performs Feature Preprocessing using chi-square and f test to remove any outliers.
4. Ranks the 6 set of features to determine which are the two top features
5. Reduce the dimensionality to two features using PCA 
6. Uses the following Machine Learning techniques, classify the three class Iris data:
    (a) Expectation Maximization
    (b) Linear Discriminant Analysis
    (c) Neural Network Method (Probabilistic NN)
    (d) Support Vector Machine

List of files in project directory
1. plots:
    - EM.jpg: Expectation Maximization plot
    - LDA.jpg: Linear Discriminant Analysis plot
    - PCA.jpg: PCA plot
2. main.py: main python file which is to be run. See instructions below for running main.py
3. ReadMe.txt: Instructions to run program and other information
4. requirements.txt: List of dependencies to run main.py
5. conclusions.txt: Conclusions based on output of program
6. logs.log: Execution results with classification results and program flow
7. iris_data_for_cleansing.csv: Raw iris data
8. tests: test runs with input folder having iris data in csv and output folder having the plots and results

Instructions to run program:
1. Open terminal and open project root directory
2. This program was developed in python 3. To install dependencies, execute below command
    pip3 install -r requirements.txt
3. Go to project root directory root folder and execute below command to run main file
    python -W ignore main.py
4. View the classification results for PNN and SVM in logs file and the three plots LDA, PCA and EM