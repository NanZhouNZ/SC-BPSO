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

def BPSO(dataset, iters, n_particles, fx, isSaveFile):
    print('X,y',X.shape, y.shape)
    particleScore = list()
    particleSize = list()
    # Initialize swarm, arbitrary
    options = {'c1': 1.49445, 'c2': 1.49445, 'w':0.729, 'k': 20, 'p':2}
    # Call instance of PSO
    dimensions = X.shape[1] # dimensions should be the number of features
    print('dimensions',dimensions)
    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dimensions, options=options)
    # Perform optimization
    cost, pos = optimizer.optimize(fx, iters=iters)
    print('cost pos',cost,pos,len(pos))
    counter = 0
    for i in pos:
        counter+=i
    print('dataset.shape in BPSO',dataset.shape)
    datasetValueList = dataset.values.tolist()
    if isSaveFile:
        addedList = []
        for index in range(len(pos)):
            # attributeIndex = i[0]
            if pos[index] == 1:
                tmpAtrribute = [j[index] for j in datasetValueList]
                addedList.append(tmpAtrribute)

        tranposedList = list(map(list, zip(*addedList)))##转置二维矩阵
        df = pandas.DataFrame(tranposedList)
        df.insert(df.shape[1], 'class', y)
        try:
            df.to_csv(curPath + '/BPSO_k'+str(counter)+'_iters_'+str(iters)+str(classifier)+time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())+'.csv', mode='w', header=False, encoding='utf_8_sig',index=False)
        except PermissionError:
            print('权限错误')
    print('df.shape',df.shape)
    iterations = list(range(1,len(optimizer.cost_history)+1))
    plt.figure(figsize=(10,7),dpi=300)
    plt.xlabel('Iterations')
    plt.ylabel('Features')
    plt.plot(iterations, optimizer.mean_pbest_history, 'r', label='pBest') 
    plt.plot(iterations, optimizer.cost_history, 'b', label='cost') 
    plt.legend()
    plt.grid(True)
    plt.savefig(curPath+'/BPSO_k'+str(counter)+'_iters_'+str(iters)+str(classifier)+".png", format="PNG")
    plt.show()

def f_per_particle(m, alpha):
    """Computes for the objective function per particle

        Inputs
        ------
        m : numpy.ndarray
            Binary mask that can be obtained from BinaryPSO, will
            be used to mask features.
        alpha: float (default is 0.5)
            Constant weight for trading-off classifier performance
            and number of features

        Returns
        -------
        numpy.ndarray
            Computed objective function
        """
    total_features = 15
    # Get the subset of the features from the binary mask
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    # Perform classification and store performance in P
    classifier.fit(X_subset, y)
    P = (classifier.predict(X_subset) == y).mean()
    return P

def f(x, alpha=0.88):
    """Higher-level method to do classification in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

def BPSO(dataset, iters, n_particles, fx, isSaveFile):
    print('X,y',X.shape, y.shape)
    particleScore = list()
    particleSize = list()
    # Initialize swarm, arbitrary
    options = {'c1': 1.49445, 'c2': 1.49445, 'w':0.729, 'k': 20, 'p':2}
    # Call instance of PSO
    dimensions = X.shape[1] # dimensions should be the number of features
    print('dimensions',dimensions)
    optimizer = ps.discrete.BinaryPSO(n_particles=n_particles, dimensions=dimensions, options=options)
    # Perform optimization
    cost, pos = optimizer.optimize(fx, iters=iters)
    print('cost pos',cost,pos,len(pos))
    counter = 0
    for i in pos:
        counter+=i
    print('dataset.shape in BPSO',dataset.shape)
    datasetValueList = dataset.values.tolist()
    if isSaveFile:
        addedList = []
        for index in range(len(pos)):
            # attributeIndex = i[0]
            if pos[index] == 1:
                tmpAtrribute = [j[index] for j in datasetValueList]
                addedList.append(tmpAtrribute)

        tranposedList = list(map(list, zip(*addedList)))##转置二维矩阵
        df = pandas.DataFrame(tranposedList)
        df.insert(df.shape[1], 'class', y)
        try:
            df.to_csv(curPath + '/BPSO_k'+str(counter)+'_iters_'+str(iters)+str(classifier)+time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())+'.csv', mode='w', header=False, encoding='utf_8_sig',index=False)
        except PermissionError:
            print('权限错误')
    print('df.shape',df.shape)
    iterations = list(range(1,len(optimizer.cost_history)+1))
    plt.figure(figsize=(10,7),dpi=300)
    plt.xlabel('Iterations')
    plt.ylabel('Features')
    plt.plot(iterations, optimizer.mean_pbest_history, 'r', label='pBest') 
    plt.plot(iterations, optimizer.cost_history, 'b', label='cost') 
    plt.legend()
    plt.grid(True)
    plt.savefig(curPath+'/BPSO_k'+str(counter)+'_iters_'+str(iters)+str(classifier)+".png", format="PNG")
    plt.show()

    return df 
