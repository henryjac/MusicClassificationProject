import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
    # Column to index for knowing which ones to specify to drop
    # danceability energy key loudness mode speechiness acousticness instrumentalness liveness valence tempo 
    # 0            1      2   3        4    5           6            7                8        9       10
    models_2_test = {
        'svc_rbf_c10_keep05':{'sk_name':SVC,'kwargs':{'C':10,'kernel':'rbf'},'preprocessing':{'keep':[0,5]}},
        'svc_rbf_c1_keep05':{'sk_name':SVC,'kwargs':{'C':1,'kernel':'rbf'},'preprocessing':{'keep':[0,5]}},
        'svc_rbf_c01_keep05':{'sk_name':SVC,'kwargs':{'C':0.1,'kernel':'rbf'},'preprocessing':{'keep':[0,5]}},
        'svc_lin_c10_keep05':{'sk_name':SVC,'kwargs':{'C':10,'kernel':'linear'},'preprocessing':{'keep':[0,5]}},
        'svc_lin_c1_keep05':{'sk_name':SVC,'kwargs':{'C':1,'kernel':'linear'},'preprocessing':{'keep':[0,5]}},
        'svc_lin_c01_keep05':{'sk_name':SVC,'kwargs':{'C':0.1,'kernel':'linear'},'preprocessing':{'keep':[0,5]}},
        'svc_pol_c10_keep05':{'sk_name':SVC,'kwargs':{'C':10,'kernel':'poly'},'preprocessing':{'keep':[0,5]}},
        'svc_pol_c1_keep05':{'sk_name':SVC,'kwargs':{'C':1,'kernel':'poly'},'preprocessing':{'keep':[0,5]}},
        'svc_pol_c01_keep05':{'sk_name':SVC,'kwargs':{'C':0.1,'kernel':'poly'},'preprocessing':{'keep':[0,5]}},
        'knn_n5_keep05':{'sk_name':KNeighborsClassifier,'kwargs':{'n_neighbors':5,'weights':'distance','algorithm':'auto'},'preprocessing':{'keep':[0,5]}},
        'knn_n10_keep05':{'sk_name':KNeighborsClassifier,'kwargs':{'n_neighbors':10,'weights':'distance','algorithm':'auto'},'preprocessing':{'keep':[0,5]}},
        'knn_n15_keep05':{'sk_name':KNeighborsClassifier,'kwargs':{'n_neighbors':15,'weights':'distance','algorithm':'auto'},'preprocessing':{'keep':[0,5]}},
        'lda_keep05':{'sk_name':LinearDiscriminantAnalysis,'kwargs':{},'preprocessing':{'keep':[0,5]}},
        'qda_reg1_keep05':{'sk_name':QuadraticDiscriminantAnalysis,'kwargs':{"reg_param":1},'preprocessing':{'keep':[0,5]}},
        'qda_reg01_keep05':{'sk_name':QuadraticDiscriminantAnalysis,'kwargs':{"reg_param":0.1},'preprocessing':{'keep':[0,5]}},
        'qda_reg001_keep05':{'sk_name':QuadraticDiscriminantAnalysis,'kwargs':{"reg_param":0.01},'preprocessing':{'keep':[0,5]}},
        'rfc_nest50_keep05':{'sk_name':RandomForestClassifier,'kwargs':{'criterion':'entropy','n_estimators':50,'max_features':'sqrt'},'preprocessing':{'keep':[0,5]}},
        'rfc_nest100_keep05':{'sk_name':RandomForestClassifier,'kwargs':{'criterion':'entropy','n_estimators':100,'max_features':'sqrt'},'preprocessing':{'keep':[0,5]}},
        'rfc_nest200_keep05':{'sk_name':RandomForestClassifier,'kwargs':{'criterion':'entropy','n_estimators':200,'max_features':'sqrt'},'preprocessing':{'keep':[0,5]}},
        'svc_rbf_c10_drop248':{'sk_name':SVC,'kwargs':{'C':10,'kernel':'rbf'},'preprocessing':{'drop':[2,4,8]}},
        'svc_rbf_c1_drop248':{'sk_name':SVC,'kwargs':{'C':1,'kernel':'rbf'},'preprocessing':{'drop':[2,4,8]}},
        'svc_rbf_c01_drop248':{'sk_name':SVC,'kwargs':{'C':0.1,'kernel':'rbf'},'preprocessing':{'drop':[2,4,8]}},
        'svc_lin_c10_drop248':{'sk_name':SVC,'kwargs':{'C':10,'kernel':'linear'},'preprocessing':{'drop':[2,4,8]}},
        'svc_lin_c1_drop248':{'sk_name':SVC,'kwargs':{'C':1,'kernel':'linear'},'preprocessing':{'drop':[2,4,8]}},
        'svc_lin_c01_drop248':{'sk_name':SVC,'kwargs':{'C':0.1,'kernel':'linear'},'preprocessing':{'drop':[2,4,8]}},
        'svc_pol_c10_drop248':{'sk_name':SVC,'kwargs':{'C':10,'kernel':'poly'},'preprocessing':{'drop':[2,4,8]}},
        'svc_pol_c1_drop248':{'sk_name':SVC,'kwargs':{'C':1,'kernel':'poly'},'preprocessing':{'drop':[2,4,8]}},
        'svc_pol_c01_drop248':{'sk_name':SVC,'kwargs':{'C':0.1,'kernel':'poly'},'preprocessing':{'drop':[2,4,8]}},
        'knn_n5_drop248':{'sk_name':KNeighborsClassifier,'kwargs':{'n_neighbors':5,'weights':'distance','algorithm':'auto'},'preprocessing':{'drop':[2,4,8]}},
        'knn_n10_drop248':{'sk_name':KNeighborsClassifier,'kwargs':{'n_neighbors':10,'weights':'distance','algorithm':'auto'},'preprocessing':{'drop':[2,4,8]}},
        'knn_n15_drop248':{'sk_name':KNeighborsClassifier,'kwargs':{'n_neighbors':15,'weights':'distance','algorithm':'auto'},'preprocessing':{'drop':[2,4,8]}},
        'lda_drop248':{'sk_name':LinearDiscriminantAnalysis,'kwargs':{},'preprocessing':{'drop':[2,4,8]}},
        'qda_reg1_drop248':{'sk_name':QuadraticDiscriminantAnalysis,'kwargs':{"reg_param":1},'preprocessing':{'drop':[2,4,8]}},
        'qda_reg01_drop248':{'sk_name':QuadraticDiscriminantAnalysis,'kwargs':{"reg_param":0.1},'preprocessing':{'drop':[2,4,8]}},
        'qda_reg001_drop248':{'sk_name':QuadraticDiscriminantAnalysis,'kwargs':{"reg_param":0.01},'preprocessing':{'drop':[2,4,8]}},
        'rfc_nest50_drop248':{'sk_name':RandomForestClassifier,'kwargs':{'criterion':'entropy','n_estimators':50,'max_features':'sqrt'},'preprocessing':{'drop':[2,4,8]}},
        'rfc_nest100_drop248':{'sk_name':RandomForestClassifier,'kwargs':{'criterion':'entropy','n_estimators':100,'max_features':'sqrt'},'preprocessing':{'drop':[2,4,8]}},
        'rfc_nest200_drop248':{'sk_name':RandomForestClassifier,'kwargs':{'criterion':'entropy','n_estimators':200,'max_features':'sqrt'},'preprocessing':{'drop':[2,4,8]}},
    }

    X_test_pre_preprocessing = pd.read_csv('data/project_test.csv', encoding='utf-8')
    latest_acc = pd.read_csv("labels/accuracies_latest")
    # with open('labels/accuracies_latest', 'w') as f:
    #     f.write('model,mean,standard deviation\n')
    for model,info in models_2_test.items():
        if latest_acc["model"].isin([model]).any():
            continue
        print(model)

        # Preprocessing
        train_data, X_test = preprocessing.preprocessing(X_test_pre_preprocessing, **info['preprocessing'])
        X_train = train_data.drop('Label',axis=1)
        y_train = train_data['Label']

        y_test = models.ML_model_prediction(
            X_train, X_test, y_train,
            info['sk_name'], **info['kwargs']
        )
        y_test = [int(x) for x in y_test]
        y_test = np.array(y_test)

        # Save the labels to a file
        y_test.tofile(f'labels/{model}_labels.csv',sep=',')

        # Test the accuracy so we can choose the one with best accuracy
        acc = np.array([])
        for i in range(100):
            X_train_acc, X_test_acc, y_train_acc, y_test_acc= train_test_split(
                X_train, y_train, test_size=0.3
            )
            acc = np.append(
                acc,
                models.get_accuracy(
                    X_train_acc, X_test_acc, y_train_acc, y_test_acc,
                    info['sk_name'], **info['kwargs']
                )
            )
        avg_acc = acc.mean()
        std_acc = acc.std()
        to_concat = pd.DataFrame([[model,avg_acc,std_acc]],columns=["model","mean","standard deviation"])
        latest_acc = pd.concat([latest_acc, to_concat])

    latest_acc.to_csv("labels/accuracies_latest")

if __name__ == '__main__':
    main()
