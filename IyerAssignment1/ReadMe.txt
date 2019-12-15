This program is developed in python3 by 'Sandesh Iyer' and does the following
1. Implements an algorithm to read in the Iris dataset
2. Implements an algorithm to visually see two sets of features and the class they belong to
3. Implements the sorting algorithm developed in Homework Problem 5

List of files in project directory
1. tests folder: Consists of two subfolders 'input' and 'output'
    a. 'input' folder contains the 'iris.arff' file with the dataset information
    b. 'output' folder has two subfolders 
        - 'before_sort_visuals': This shows the plots of each feature vs points for each class without sorting (4 features)
        - 'after_sort_visuals': This shows the plots of each feature vs points for each class after using sorting algorithm (4 features)
2. main.py: main python file which is to be run to produce output results in 'output' folder of project main directory. See instructions below for running main.py
3. ReadMe.txt: Instructions to run program and other information
4. requirements.txt: List of dependencies to run main.py
5. conclusions.txt: Conclusions based on output of program
6. logs.log: Execution results with print out statements

Instructions to run program:
1. Open terminal and open project root directory
2. This program was developed in python 3. To install dependencies, execute below command
    pip3 install -r requirements.txt
3. Go to project root directory root folder and execute below command to run main file
    python main.py
4. Plots generated in 'output' folder 

Compiler/IDE: Python3. Code developed on Visual Studio Code IDE.