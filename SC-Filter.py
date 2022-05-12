#coding=utf-8
# from scipy.sparse import data
from scipy.sparse import data
from scipy.stats.stats import spearmanr
from scipy.stats.stats import spearmanr
from sklearn.feature_selection import chi2
import os
import pandas
from numpy.core.fromnumeric import mean
import numpy as np
from pandas.core.frame import DataFrame
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from math import log

import matplotlib.pyplot as plt #MatPlotLib usado para desenhar o gráfico criado com o NetworkX
# Import PySwarms
import pyswarms as ps

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

# import datetime as dt
from pyswarms.utils.functions import single_obj as single_obj_func
from pyswarms.single import global_best as global_best_func
from pyswarms.utils.search import grid_search
curPath = os.getcwd()
import time

global classifier
global classifierList 
    
rf_c = RandomForestClassifier(criterion="entropy", random_state=0)
svm_c = svm.SVC(kernel='linear', C=1, random_state=1)
knn_c = KNeighborsClassifier(n_neighbors=1)
j48_c = tree.DecisionTreeClassifier(random_state=1)

classifierList = [rf_c, svm_c, knn_c, j48_c]

# global X
global y
def read_dataset(filename):
    fr = open(filename, 'r')
    all_lines = fr.readlines()  # list形式,每行为1个str
    labels = []
    labelCounts = {}
    dataset = []
    for line in all_lines[0:]:
        line = line.strip().split(',')  # 以逗号为分割符拆分列表
        dataset.append(line)
    return dataset, labels

def calcuChi2(X, y):
    return list(chi2(X, y))[0]

def calcuSpearman(X, y):
    spearmanValueList = []
    from scipy import stats
    print('X.shape[1]', X.shape[1])
    for i in range(X.shape[1]):
        col = X.iloc[:,i]
        spearmanValueList.append(abs(stats.spearmanr(col, y)[0]))
    return spearmanValueList

def evaluationFunc(X,y):
    chi2ValueList = calcuChi2(X, y)
    spearmanValueList = calcuSpearman(X, y)
    scoreDict = {}

    chi2ValueList = normalizationList(chi2ValueList)
    spearmanValueList = normalizationList(spearmanValueList)

    print(len(chi2ValueList), len(spearmanValueList))


    for i in range(len(chi2ValueList)):
        scoreDict[i] = spearmanValueList[i] + (1/chi2ValueList[i])

    scoreSortedList = sorted(scoreDict.items(), key=lambda item:item[1], reverse=True)
    return scoreSortedList

def normalizationList(origList):
    normalizedList = []
    maxValue = max(origList)
    for i in origList:
        normalizedValue = i/maxValue
        normalizedList.append(normalizedValue)
    return normalizedList