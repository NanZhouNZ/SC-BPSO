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


rf_c = RandomForestClassifier(criterion="entropy", random_state=0)
svm_c = svm.SVC(kernel='linear', C=1, random_state=1)
knn_c = KNeighborsClassifier(n_neighbors=1)
j48_c = tree.DecisionTreeClassifier(random_state=1)
classifierList = [rf_c, svm_c, knn_c, j48_c]

def predictWithLeaveOneOut(clf, dataset, fileName, isSave):
    clfName = ''
    if  isinstance(clf, svm.SVC):
        clfName = 'svm'
    elif isinstance(clf, tree.DecisionTreeClassifier):
        clfName = 'DecisionTree'
    elif isinstance(clf, RandomForestClassifier):
        clfName = 'RandomForest'
    elif isinstance(clf, KNeighborsClassifier):
        clfName = 'knn'
    resultText = clfName
    df = pandas.DataFrame(dataset)
    x = df.iloc[:,:-1].astype('float')
    y = df.iloc[:,-1].astype('float')
    loo = LeaveOneOut()
    predictList = []
    for train_index,test_index in loo.split(dataset):
        x_train, x_test = DataFrame(x, index=train_index), DataFrame(x, index=test_index)
        y_train, y_test = DataFrame(y, index=train_index), DataFrame(y, index=test_index)
        score = clf.fit(x_train, y_train.values.ravel())
        predictList.append(clf.score(x_test, y_test.values.ravel()))
    dimensions = dataset.shape[1]-1
    precision = mean(predictList)
    print(clfName, dimensions, precision)
    if clfName == 'svm':
        SVMDict[dimensions] = mean(predictList)
    elif clfName == 'DecisionTree':
        DecisionTreeDict[dimensions] = mean(predictList)
    elif clfName == 'RandomForest':
        RandomForestDict[dimensions] = mean(predictList)
    elif clfName == 'knn':
        KNNDict[dimensions] = mean(predictList)
    
    resultText = resultText + fileName + ':' + str(mean(predictList))
    if isSave:
        with open(curPath + fileName.split('_')[0] +'_Sum'+'.csv', 'a+', encoding='utf-8') as wf:
            wf.write(resultText+'\n')

def classWith4Func(dataset, filename, isSave):
    clfList = [rf_c, svm_c, knn_c, j48_c]
    for clf in clfList:
        predictWithLeaveOneOut(clf, dataset, filename, isSave)
