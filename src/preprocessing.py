import random
import sklearn

import pandas as pd
import statistics as s
import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt
from scipy.optimize import minimize
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC # Supprort Vector classifier
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('src')
import models

def preprocessing(X_test=None):
    """
    Preprocesses the project_train.csv file by removing irrelevant columns
    and rows.
    Returns the DataFrame object
    """
    train_file = 'data/project_train.csv'
    data = pd.read_csv(train_file, encoding='utf-8')
    data = pd.DataFrame(data)

    # Remove rows with faulty values, which
    # we found in the labels 'energy' and 'loudness'
    energy_idx_err = data.idxmax()['energy']
    loudness_idx_err = data.idxmin()['loudness']
    data = data.drop([energy_idx_err, loudness_idx_err], axis=0)

    # Drop columns 'danceability', 'valence'
    # as their correlation with the Label is low
    # Also drop column 'instrumentalness' as it's highyl correlated with
    # 'acousticness'
    to_drop = ['mode','key','liveness']
    data = data.drop(to_drop, axis=1)
    data = normalize(data) # Use deviation of mean to normalize
    if X_test is not None:
        X_test = X_test.drop(to_drop, axis=1)
        X_test = normalize(X_test)
        return data, X_test
    return data

def normalize(dataframe):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

def best_features_from_RFC(data, n_best=4):
    """
    Returns a DataFrame with the `n_best` number of best features
    gathered from RandomForestClassifier model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3
    )
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    model.fit(X_train, y_train)

    features = data.drop('Label', axis=1).columns
    importances = model.feature_importances_
    indices = np.argsort(importances)

    return data.drop(features[indices[:-4]], axis=1)

def test_accuracy():
    # ------------------- IMPORT DATA ----------------
    # Import data
    svc_accs = np.array([])
    knn_accs = np.array([])
    lda_accs = np.array([])
    qda_accs = np.array([])
    rfc_accs = np.array([])
    # data = preprocessing()
    train_file = 'data/project_train.csv'
    data = pd.read_csv(train_file, encoding='utf-8')
    for i in range(10):
        # Split data
        # data = best_features_from_RFC(data)
        X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3
        )
        to_drop = ['key','mode']
        X_train = X_train.drop(to_drop, axis=1)
        X_test = X_test.drop(to_drop, axis=1)

        acc = models.get_accuracy(
            X_train, X_test, y_train, y_test,
            SVC, C=0.2, kernel='rbf'
        )
        svc_accs = np.append(rfc_accs, acc)

        acc = models.get_accuracy(
            X_train, X_test, y_train, y_test,
            KNeighborsClassifier, n_neighbors=10, weights='distance', algorithm='auto'
        )
        knn_accs = np.append(rfc_accs, acc)

        acc = models.get_accuracy(
            X_train, X_test, y_train, y_test,
            LinearDiscriminantAnalysis
        )
        lda_accs = np.append(rfc_accs, acc)

        acc = models.get_accuracy(
            X_train, X_test, y_train, y_test,
            QuadraticDiscriminantAnalysis, reg_param=0.01
        )
        qda_accs = np.append(rfc_accs, acc)

        acc = models.get_accuracy(
            X_train, X_test, y_train, y_test,
            RandomForestClassifier, criterion='entropy', n_estimators=200, max_features='sqrt'
        )
        rfc_accs = np.append(rfc_accs, acc)
    avg_acc_rfc = rfc_accs.sum()/rfc_accs.size
    max_acc_rfc = rfc_accs.max()
    accuracies = [svc_accs, knn_accs, lda_accs, qda_accs, rfc_accs]
    accuracies_name = ['SVC','KNN','LDA','QDA','RFC']
    for accs,name in zip(accuracies,accuracies_name):
        avg_acc = accs.sum()/accs.size
        max_acc = accs.max()
        print(f'{name}:')
        print(f'Average accuracy: {avg_acc}\nMax accuracy: {max_acc}')

def main():
    test_accuracy()

if __name__ == '__main__':
    main()
