import numpy as np
import pandas as pd
from sklearn.svm import SVC # Supprort Vector classifier

import sys
sys.path.append('src')
import preprocessing
import models

def main():
    X_test = pd.read_csv('data/project_test.csv', encoding='utf-8')
    train_data = pd.read_csv('data/project_train.csv', encoding='utf-8')
    train_data, X_test = preprocessing.preprocessing(X_test)

    X_train = train_data.drop('Label',axis=1)
    y_train = train_data['Label']
    print(y_train.sum()/y_train.shape[0])

    y_test = models.ML_model_prediction(
        X_train, X_test, y_train,
        SVC, C=10, kernel='rbf'
    )
    y_test = [int(x) for x in y_test]
    y_test = np.array(y_test)

    # acc = y_test - y_train
    # acc = 1 - np.dot(acc,acc) / len(y_test)
    # print(acc)
    print(y_test)
    print(y_test.sum()/y_test.shape[0])

    y_test.tofile('data/labels.csv',sep=',')

if __name__ == '__main__':
    main()
