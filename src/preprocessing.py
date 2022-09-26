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

def preprocessing():
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
    loudness_idx_err = data.idxmax()['loudness']
    data = data.drop([energy_idx_err, loudness_idx_err], axis=0)

    # Drop columns 'danceability', 'valence'
    # as their correlation with the Label is low
    # Also drop column 'instrumentalness' as it's highyl correlated with
    # 'acousticness'
    data = data.drop(['danceability','valence','instrumentalness'], axis=1)

    return data

def test_accuracy():
    # ------------------- IMPORT DATA ----------------
    # Import data
    rfc_accs = np.array([])
    data = preprocessing()
    for i in range(100):
        # data = pd.read_csv (r'data/project_train.csv', encoding='utf-8')
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3
        )

        # acc = models.get_accuracy(
        #     X_train, X_test, y_train, y_test,
        #     SVC, C=1, kernel='linear'
        # )
        # print('Accuracy with SVM')
        # print(acc)

        # acc = models.get_accuracy(
        #     X_train, X_test, y_train, y_test,
        #     KNeighborsClassifier, n_neighbors=10, weights='distance', algorithm='auto'
        # )
        # print('Accuracy with KNN')
        # print(acc)

        # acc = models.get_accuracy(
        #     X_train, X_test, y_train, y_test,
        #     LinearDiscriminantAnalysis
        # )
        # print('Accuracy with LDA')
        # print(acc)

        # acc = models.get_accuracy(
        #     X_train, X_test, y_train, y_test,
        #     QuadraticDiscriminantAnalysis, reg_param=0.01
        # )
        # print('Accuracy with QDA')
        # print(acc)

        acc = models.get_accuracy(
            X_train, X_test, y_train, y_test,
            RandomForestClassifier, criterion='entropy', n_estimators=200, max_features='sqrt'
        )
        # print('Accuracy with rfc')
        # print(acc)
        rfc_accs = np.append(rfc_accs, acc)
    avg_acc = rfc_accs.sum()/rfc_accs.size
    max_acc = rfc_accs.max()
    print(f'Average accuracy: {avg_acc}\nMax accuracy: {max_acc}')

def main():
    test_accuracy()

if __name__ == '__main__':
    main()
