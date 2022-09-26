import pandas as pd
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statistics as s
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from sklearn.svm import SVC # Supprort Vector classifier
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('src')
import models

def main():
    # ------------------- IMPORT DATA ----------------
    data = pd.read_csv (r'data/project_train.csv', encoding='utf-8') # Import data
    inputs = pd.DataFrame(data).to_numpy()
    print('size of data input')
    print(np.shape(inputs))
    np.random.shuffle(inputs)
    # -----------------  TRAINING DATA ---------------------------------
    X_train = inputs[0:374,0:11] # training data (without the labels)
    y_train = inputs[0:374,11] # training data (only labels)
    # ----------------- EVALUATION DATA ---------------------------------
    X_test = inputs[375:,0:11] # Using last 25% of data to evaluate how good the classifier is.
    y_test = inputs[375:,11] # correct labels on the evaluation data.
    print('-----')

    acc = models.get_accuracy(
        X_train, X_test, y_train, y_test,
        SVC, C=1, kernel='linear'
    )
    print('Accuracy with SVM')
    print(acc)

    acc = models.get_accuracy(
        X_train, X_test, y_train, y_test,
        KNeighborsClassifier, n_neighbors=10, weights='distance', algorithm='auto'
    )
    print('Accuracy with KNN')
    print(acc)

    acc = models.get_accuracy(
        X_train, X_test, y_train, y_test,
        LinearDiscriminantAnalysis
    )
    print('Accuracy with LDA')
    print(acc)

    acc = models.get_accuracy(
        X_train, X_test, y_train, y_test,
        QuadraticDiscriminantAnalysis, reg_param=0.01
    )
    print('Accuracy with QDA')
    print(acc)

    acc = models.get_accuracy(
        X_train, X_test, y_train, y_test,
        RandomForestClassifier, criterion='entropy', n_estimators=200, max_features='sqrt'
    )
    print('Accuracy with rfc')
    print(acc)

if __name__ == '__main__':
    main()
