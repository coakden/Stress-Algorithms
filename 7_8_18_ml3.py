#Detecting Word Stress with Decision Trees and Support Vector Machines
#
#Author: Chris Oakden
#
#
# This script uses decision tree classifiers and SVMs (from scikit learn) to predict
# word stress assignment in Lithuanian from five acoustic measures: duration,
# fundamental frequency, F1, F2, and intensity. Functions are defined for each type
# of classifier, and for each acoustic correlate; these functions partition the dataset
# randomly into training data and test data (90-10), offer predictions for stress
# assignment, and report the accuracy of the predictions. Functions take a single
# argument which specifies the number of test iterations.
#
# Two additional functions (one for each classifier type) aggregate the test results
# into a dictionary comparing mean accuracy of tests across acoustic correlates. This
# allows for a broad comparison of which acoustic correlates were more accurate
# predictors of word stress assignment.
################

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn import svm 

u_dat = pd.read_csv("u.csv")

#to check the names of the columns

columns = u_dat.columns.values

#assign acoustic measure columns to variables

X = u_dat["NormalizedDuration"]

Z = u_dat["f0_midpoint"]

V = u_dat["F1_midpoint"]

H = u_dat["F2_midpoint"]

I = u_dat["intensity_midpoint"]

#change 'stress' to an integer, and convert it into a dataframe
#(when using pandas, you need to do this for sklearn to recognize it)


y = u_dat["Stress"].map({'Y':1, 'N':0}).to_frame()

#use ravel() instead of to_frame() for the y coordinates when using an
#svm, otherwise you get the error message:
#   DataConversionWarning: A column-vector y was passed when a 1d array
#   was expected. Please change the shape of y to (n_samples, ), for
#   example using ravel()

y2 = u_dat["Stress"].map({'Y':1, 'N':0}).ravel()

#iterate the experiments multiple times using different partitions
#start by creating a counter and a list and iterate over the
#counter (which is specified as an argument to the function), printing out
#the accuracy score of each test which then populates the list

#create a unique function for each acoustic correlate

###decision tree algorithms###

def run_treetest_dur(num):
    n = 0
    l = []
    for n in range(num):
        X_train, X_test, y_train, y_test = tts(X, y, test_size = .1)
        my_c1 = tree.DecisionTreeClassifier()
        my_c1.fit(X_train.values.reshape(-1,1), y_train)
        predictions1 = my_c1.predict(X_test.values.reshape(-1,1))
        score = accuracy_score(y_test, predictions1)
        l.append(score)
        n += 1
    return l

def run_treetest_f0(num):
    n = 0
    l = []
    for n in range(num):
        Z_train, Z_test, y_train, y_test = tts(Z, y, test_size = .1)
        my_c1 = tree.DecisionTreeClassifier()
        my_c1.fit(Z_train.values.reshape(-1,1), y_train)
        predictions1 = my_c1.predict(Z_test.values.reshape(-1,1))
        score = accuracy_score(y_test, predictions1)
        l.append(score)
        n += 1
    return l

def run_treetest_f1(num):
    n = 0
    l = []
    for n in range(num):
        V_train, V_test, y_train, y_test = tts(V, y, test_size = .1)
        my_c1 = tree.DecisionTreeClassifier()
        my_c1.fit(V_train.values.reshape(-1,1), y_train)
        predictions1 = my_c1.predict(V_test.values.reshape(-1,1))
        score = accuracy_score(y_test, predictions1)
        l.append(score)
        n += 1
    return l

def run_treetest_f2(num):
    n = 0
    l = []
    for n in range(num):
        H_train, H_test, y_train, y_test = tts(H, y, test_size = .1)
        my_c1 = tree.DecisionTreeClassifier()
        my_c1.fit(H_train.values.reshape(-1,1), y_train)
        predictions1 = my_c1.predict(H_test.values.reshape(-1,1))
        score = accuracy_score(y_test, predictions1)
        l.append(score)
        n += 1
    return l

def run_treetest_int(num):
    n = 0
    l = []
    for n in range(num):
        I_train, I_test, y_train, y_test = tts(I, y, test_size = .1)
        my_c1 = tree.DecisionTreeClassifier()
        my_c1.fit(I_train.values.reshape(-1,1), y_train)
        predictions1 = my_c1.predict(I_test.values.reshape(-1,1))
        score = accuracy_score(y_test, predictions1)
        l.append(score)
        n += 1
    return l

#########################

###svm (support vector machines)###

def run_svmtest_dur(num):
    n = 0
    l = []
    for n in range(num):
        X_train, X_test, y2_train, y2_test = tts(X, y2, test_size = .1)
        my_c1 = svm.SVC()
        my_c1.fit(X_train.values.reshape(-1,1), y2_train)
        predictions1 = my_c1.predict(X_test.values.reshape(-1,1))
        score = accuracy_score(y2_test, predictions1)
        l.append(score)
        n += 1
    return l

def run_svmtest_f0(num):
    n = 0
    l = []
    for n in range(num):
        Z_train, Z_test, y2_train, y2_test = tts(Z, y2, test_size = .1)
        my_c1 = svm.SVC()
        my_c1.fit(Z_train.values.reshape(-1,1), y2_train)
        predictions1 = my_c1.predict(Z_test.values.reshape(-1,1))
        score = accuracy_score(y2_test, predictions1)
        l.append(score)
        n += 1
    return l

def run_svmtest_f1(num):
    n = 0
    l = []
    for n in range(num):
        V_train, V_test, y2_train, y2_test = tts(V, y2, test_size = .1)
        my_c1 = svm.SVC()
        my_c1.fit(V_train.values.reshape(-1,1), y2_train)
        predictions1 = my_c1.predict(V_test.values.reshape(-1,1))
        score = accuracy_score(y2_test, predictions1)
        l.append(score)
        n += 1
    return l

def run_svmtest_f2(num):
    n = 0
    l = []
    for n in range(num):
        H_train, H_test, y2_train, y2_test = tts(H, y2, test_size = .1)
        my_c1 = svm.SVC()
        my_c1.fit(H_train.values.reshape(-1,1), y2_train)
        predictions1 = my_c1.predict(H_test.values.reshape(-1,1))
        score = accuracy_score(y2_test, predictions1)
        l.append(score)
        n += 1
    return l

def run_svmtest_int(num):
    n = 0
    l = []
    for n in range(num):
        I_train, I_test, y2_train, y2_test = tts(I, y2, test_size = .1)
        my_c1 = svm.SVC()
        my_c1.fit(I_train.values.reshape(-1,1), y2_train)
        predictions1 = my_c1.predict(I_test.values.reshape(-1,1))
        score = accuracy_score(y2_test, predictions1)
        l.append(score)
        n += 1
    return l



#create a function which creates a dictionary with the results from each test
#use the dict and zip functions to create a dictionary; much better than trying
#to iterate over some value (like iter(len(keys)) or something which won't
#give you the key values, but rather just numbers)

def tree_tests(x):
    keys = ['Duration', 'f0', 'F1', 'F2', 'Intensity']
    values = [np.mean(run_treetest_dur(x)), np.mean(run_treetest_f0(x)),
              np.mean(run_treetest_f1(x)), np.mean(run_treetest_f2(x)),
              np.mean(run_treetest_int(x))]
    dictionary = dict(zip(keys, values))
    print(dictionary)

def svm_tests(x):
    keys = ['Duration', 'f0', 'F1', 'F2', 'Intensity']
    values = [np.mean(run_svmtest_dur(x)), np.mean(run_svmtest_f0(x)),
              np.mean(run_svmtest_f1(x)), np.mean(run_svmtest_f2(x)),
              np.mean(run_svmtest_int(x))]
    dictionary = dict(zip(keys, values))
    print(dictionary)

