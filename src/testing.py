import numpy as np
import pandas as pd

from scipy.optimize import minimize
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC # Supprort Vector classifier
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

import sys,os
sys.path.append('src')
import models

def test_accuracy(test_size=0.3):
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
            data.iloc[:, :-1], data.iloc[:, -1], test_size=test_size
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

def best_model_acc_latest():
    df = pd.read_csv('labels/accuracies_latest')
    print('The model with highest accuracy is: {}'.format(df.loc[df["mean"].idxmax()]["model"]))
    print('The model with accuracy over 0.8 and lowest standard deviation is {}'.format(
        df.loc[df[df["mean"] > 0.8]["standard deviation"].idxmin()]["model"])
    )

def main():
    test_accuracy()

if __name__ == '__main__':
    main()
