import pandas as pd
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statistics as s
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold, train_test_split

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
    # Import data
    data = pd.read_csv (r'data/project_train.csv', encoding='utf-8')
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3
    )

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
