#Detecting Word Stress with a Duration-Based Algorithm
#
#Author: Chris Oakden
#
#
# This script defines several functions which predict stress in a set of
# trisyllabic Lithuanian words. Words from the dataset are from an elicitation
# of nominal forms with stress on either the penult or ultima. These functions
# compare vowel durations within each word, and predict stress to fall on the
# syllable which is 20% longer than the other two syllables. If no syllable is
# 20% longer than the other two, no stress is predicted. Another function
# compares the predictions to the actual stress assignment. A final function
# calculates the accuracy of the algorithm in predicting stress.
######################

import numpy as np
import pandas as pd
import statistics

data = pd.read_csv("F2Durs.csv")

def predict(data):
    p = []
    data = data
    for row in range(len(data)):
        v1 = data.loc[row, 'V1D']
        v2 = data.loc[row, 'V2D']
        v3 = data.loc[row, 'V3D']
        if ((v1 - v2)/v2)*100 >= 20 and ((v1 - v3)/v3)*100 >= 20:
            p.append(1)
        elif ((v2 - v1)/v1)*100 >= 20 and ((v2 - v3)/v3)*100 >= 20:
            p.append(2)
        elif ((v3 - v1)/v1)*100 >= 20 and ((v3 - v2)/v2)*100 >= 20:
            p.append(3)
        else:
            p.append(0)
    return p

def predict_vs_stress(data):
    p = []
    data = data
    for row in range(len(data)):
        v1 = data.loc[row, 'V1D']
        v2 = data.loc[row, 'V2D']
        v3 = data.loc[row, 'V3D']
        if ((v1 - v2)/v2)*100 >= 20 and ((v1 - v3)/v3)*100 >= 20:
            p.append(str(1) + ':' + str(data.loc[row, 'Stress']))
        elif ((v2 - v1)/v1)*100 >= 20 and ((v2 - v3)/v3)*100 >= 20:
            p.append(str(2) + ':' + str(data.loc[row, 'Stress']))
        elif ((v3 - v1)/v1)*100 >= 20 and ((v3 - v2)/v2)*100 >= 20:
            p.append(str(3) + ':' + str(data.loc[row, 'Stress']))
        else:
            p.append(str(0) + ':' + str(data.loc[row, 'Stress']))
    return p

#still haven't figured out how to get this to work; want a list of strings with
# 'Predicted' and 'Actual' for clarity
def predict_vs_stress_d(data):
    p = []
    data = data
    for row in range(len(data)):
        v1 = data.loc[row, 'V1D']
        v2 = data.loc[row, 'V2D']
        v3 = data.loc[row, 'V3D']
        if ((v1 - v2)/v2)*100 >= 20 and ((v1 - v3)/v3)*100 >= 20:
            p.append('Predicted:'+ str(1) + '; Actual:' + str(data.loc[row, 'Stress']))
        elif ((v2 - v1)/v1)*100 >= 20 and ((v2 - v3)/v3)*100 >= 20:
            p.append('Predicted:'+ str(2) + '; Actual:' + str(data.loc[row, 'Stress']))
        elif ((v3 - v1)/v1)*100 >= 20 and ((v3 - v2)/v2)*100 >= 20:
            p.append('Predicted:'+ str(3) + '; Actual:' + str(data.loc[row, 'Stress']))
        else:
            p.append('Predicted:'+ str(0) + '; Actual:' + str(data.loc[row, 'Stress']))
    t = []
    for row in range(len(data)):
        t.append(data[row, 'File'])
    output = dict(zip(t, p))
    print(output)

def accuracy(data):
    p = []
    data = data
    for row in range(len(data)):
        v1 = data.loc[row, 'V1D']
        v2 = data.loc[row, 'V2D']
        v3 = data.loc[row, 'V3D']
        if ((v1 - v2)/v2)*100 >= 20 and ((v1 - v3)/v3)*100 >= 20:
            prediction = 1
        elif ((v2 - v1)/v1)*100 >= 20 and ((v2 - v3)/v3)*100 >= 20:
            prediction = 2
        elif ((v3 - v1)/v1)*100 >= 20 and ((v3 - v2)/v2)*100 >= 20:
            prediction = 3
        else:
            prediction = 0
        if prediction == data.loc[row, 'Stress']:
            p.append(1)
        else:
            p.append(0)
    total = sum(p) / len(data)
    return total

#### exploratory statistics

#what's the average?
avg1 = statistics.mean(data['V1D'])

##what are the ranges of the values for each position?
Range1 = max(data['V1D']) - min(data['V1D'])
Range2 = max(data['V2D']) - min(data['V2D'])
Range3 = max(data['V3D']) - min(data['V3D'])

### convert from seconds into milliseconds
mili1 = list(map(lambda x: x*1000, data['V1D'])
    





        


    


 
