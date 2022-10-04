import numpy as np
import pandas as pd
from sklearn.svm import SVC # Supprort Vector classifier
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('src')
import preprocessing
import models

def main():
    X_test = pd.read_csv('data/project_test.csv', encoding='utf-8')
    train_data, X_test = preprocessing.preprocessing(X_test)

    X_train = train_data.drop('Label',axis=1)
    y_train = train_data['Label']

    models_2_test = {
        'svc_rbf_c10':{'sk_name':SVC,'kwargs':{'C':10,'kernel':'rbf'}},
        'svc_rbf_c1':{'sk_name':SVC,'kwargs':{'C':1,'kernel':'rbf'}},
        'svc_rbf_c01':{'sk_name':SVC,'kwargs':{'C':0.1,'kernel':'rbf'}},
        'knn_n10':{'sk_name':KNeighborsClassifier,'kwargs':{'n_neighbors':10,'weights':'distance','algorithm':'auto'}},
        'knn_n15':{'sk_name':KNeighborsClassifier,'kwargs':{'n_neighbors':15,'weights':'distance','algorithm':'auto'}},
        'knn_n5':{'sk_name':KNeighborsClassifier,'kwargs':{'n_neighbors':5,'weights':'distance','algorithm':'auto'}},
    }
    for model,info in models_2_test.items():
        print(model)
        y_test = models.ML_model_prediction(
            X_train, X_test, y_train,
            info['sk_name'], **info['kwargs']
        )
        y_test = [int(x) for x in y_test]
        y_test = np.array(y_test)

        # Save the labels to a file
        y_test.tofile(f'labels/{model}_labels.csv',sep='\n')

if __name__ == '__main__':
    main()
