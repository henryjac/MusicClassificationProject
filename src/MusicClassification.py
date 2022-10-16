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

def loglinstep(minval, maxval):
    result = []
    for i in range (int(np.log10(minval)), int(np.log10(maxval))):
        result += [j*(10**i) for j in range(1, 10)]
    return result + [maxval]

def main():
    # Column to index for knowing which ones to specify to drop
    # danceability energy key loudness mode speechiness acousticness instrumentalness liveness valence tempo 
    # 0            1      2   3        4    5           6            7                8        9       10
    models_2_test = {
        'svc':{
            'sk_name':SVC,
            'params':{
                'C':loglinstep(0.01, 100),
                'kernel':['linear', 'rbf', 'poly']
            },
            'preprocessing':{}
        },
        'knn':{
            'sk_name':KNeighborsClassifier,
            'params':{
                'n_neighbors':[i for i in range(3, 26)],
                'weights':['distance'],
                'algorithm':['auto']
            },
            'preprocessing':{}
        },
        'lda':{
            'sk_name':LinearDiscriminantAnalysis,
            'params':{},
            'preprocessing':{}
        },
        'qda':{
            'sk_name':QuadraticDiscriminantAnalysis,
            'params':{
                'reg_param':loglinstep(0.00001, 0.1)
            },
            'preprocessing':{}
        },
        'rfc':{
            'sk_name':RandomForestClassifier,
            'params':{
                'criterion':['entropy'],
                'n_estimators':[i for i in range(5, 501, 1)],
                'max_features':['sqrt']
            },
            'preprocessing':{},
            'threshold':0.4,
        }
    }

    X_test_pre_preprocessing = pd.read_csv('data/project_test.csv', encoding='utf-8')

    drop_order = [4,2,8,9,10,1,7,6,3,5,0]
    for model, info in models_2_test.items():
        if model != 'rfc':
            continue
        best_acc = 0.0
        best_params = None
        best_drop = None

        print(f"Running search for {model}", end="", flush=True)

        for i in range(len(drop_order)):
            drop = drop_order[:i]
            info['preprocessing']['drop'] = drop

            # Preprocessing
            train_data, X_test = preprocessing.preprocessing(X_test_pre_preprocessing, **info['preprocessing'])
            X_train = train_data.drop('Label',axis=1)
            y_train = train_data['Label']

            (estimator, params, score) = models.grid_search(
                X_train, y_train,
                info['sk_name'], info['params'],
                verbose=False,
                n_cores=28,
                preprocessing=drop,
            )

            if score > best_acc:
                best_acc = score
                best_params = params
                best_drop = drop

            if model == 'rfc':
                y_test = model.predict_proba(X_test)
                y_test = (y_test[:,1] >= 0.4).astype('int')
            else:
                y_test = estimator.predict(X_test)
            labels = np.array([int(i) for i in y_test])

            preprocessing_data = ''.join([str(i) for i in info['preprocessing']['drop']])
            if preprocessing_data == '':
                preprocessing_data = ''
            else:
                preprocessing_data = f"_drop{preprocessing_data}"
            folder = model[:3]
            labels.tofile(f'labels/{folder}/{model}{preprocessing_data}_labels.csv', sep=',')
            print(".", end="", flush=True)

        print(f"\n    best score: {best_acc}")
        print(f"    with param: {best_params}")
        print(f"    when omitt: {best_drop}")

def average_labels_rfc(X_train, X_test, y_train, info):
    y_test_final = pd.DataFrame()
    tries = 50
    for i in range(tries):
        y_test = models.ML_model_prediction(
            X_train, X_test, y_train,
            info['sk_name'], **info['kwargs']
        )
        y_test = pd.DataFrame(y_test)
        y_test_final = pd.concat([y_test_final, y_test])
    df_sum = y_test_final.sum().transpose()
    df_sum[df_sum > tries/2] = 1
    df_sum[df_sum <= tries/2] = 0 # If we have less than half 0s, set to 0, otherwise 1
    return df_sum

def test_cross_validation(k=5):
    drop = ['key', 'mode']
    processed_data = preprocessing.preprocessing(drop=drop)
    X_train = np.array(processed_data.drop('Label', axis=1))
    y_train = np.array(processed_data['Label'])

    accuracy = models.cross_validate(X_train, y_train, k, SVC, C=0.2, kernel='rbf')
    print(f"Tested cross validation with k={k}: {accuracy}")

def test_grid_search():
    drop = ['key', 'mode']
    processed_data = preprocessing.preprocessing(drop=drop)
    X_train = np.array(processed_data.drop('Label', axis=1))
    y_train = np.array(processed_data['Label'])

    (estimator, params, score) = models.grid_search(
        X_train, y_train, SVC,
        {
            'C':([0.01*i for i in range(1, 10)] + [0.1*i for i in range(1, 10)]
                   + [i for i in range(1, 10)] + [10*i for i in range(1, 10)]),
            'kernel':['rbf', 'linear']
        }
    )

if __name__ == '__main__':
    #test_grid_search()
    #test_cross_validation(10)
    main()
