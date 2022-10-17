import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # Supprort Vector classifier
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

import sys
import time
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
                'C':loglinstep(0.01, 10),
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
        # 'rfc':{
        #     'sk_name':RandomForestClassifier,
        #     'params':{
        #         'criterion':['entropy'],
        #         'n_estimators':[i for i in range(5, 501, 5)],
        #         'max_features':['sqrt']
        #     },
        #     'preprocessing':{}
        # }
    }

    batches = 20
    results = {}

    # drop_order = [4,2,8,9,10,1,7,6,3,5,0]
    drop_order = [4,2,8,9,10,1]
    feature_cross = [7,6,3,5,0] # Maybe include dropped as well
    for model, info in models_2_test.items():
        # creates entry for new model
        results[model] = {}

        print(f"Running search for {model}", flush=True)

        for i in range(len(feature_cross)):
            for j in range(i+1):
                for k in range(len(drop_order)):
                    drop = drop_order[:k]
                    if feature_cross[i] is None or feature_cross[j] is None:
                        feature_crosses = []
                    else:
                        feature_crosses = [feature_cross[i],feature_cross[j]]
                    info['preprocessing']['drop'] = drop
                    info['preprocessing']['feature_crosses'] = feature_crosses
                    drop = (tuple(drop),tuple(feature_crosses))

                    print(f"    preprocessing {i},{j},{k}", end = "", flush=True)

                    # creates entry for preprocessing option and parameters
                    results[model][drop] = {}
                    for parameter in info['params']:
                        results[model][drop][parameter] = []
                    results[model][drop]['results'] = []

                    y_test = []
                    
                    for _ in range(batches):
                        # Preprocessing
                        train_data, X_test = preprocessing.preprocessing(**info['preprocessing'])
                        X_train = train_data.drop('Label',axis=1)
                        y_train = train_data['Label']

                        (estimator, params, score) = models.grid_search(
                            X_train, y_train,
                            info['sk_name'], info['params'],
                            verbose=False,
                            n_cores=30,
                            preprocessing=drop,
                        ) # this takes time

                        for param in params:
                            results[model][drop][param] += [params[param]]
                        results[model][drop]['results'] += [score]

                        y_temp = estimator.predict(X_test)
                        y_test += [[int(i) for i in y_temp]]
                        
                        print(".", end="", flush=True)
                    print()

                    # voting
                    labels = [sum([y_test[j][i] for j in range(batches)]) for i in range(len(y_test[0]))]
                    labels = np.array([int(np.round(labels[i]/batches)) for i in range(len(y_test[0]))])

                    preprocessing_data = ''.join([str(i) for i in info['preprocessing']['drop']])
                    feature_cross_data = ''.join([str(i) for i in feature_crosses])
                    preprocessing_data = '' if preprocessing_data == '' else f"_drop{preprocessing_data}"
                    feature_cross_data = '' if feature_cross_data == '' else f"_cross{feature_cross_data}"
                    folder = model[:3]
                    labels.tofile(f'labels/{folder}/{model}{preprocessing_data}{feature_cross_data}_labels.csv', sep=',')

    # data analysis
    print(f"Analysing results from {batches} batches...")
    statistics = {}
    for model, model_results in results.items():
        best_acc = 0.0
        best_drop = None

        statistics[model] = {}
        
        for preopts, preopts_results in model_results.items():
            point_statistics = {}
            for value, values in preopts_results.items():
                if isinstance(values[0], str):
                    point_statistics[value] = [values[0]]
                    continue
                npify = np.array(values)
                point_statistics[value] = [
                    npify.mean(),
                    npify.std(),
                    np.median(npify)
                ]

            if point_statistics['results'][0] > best_acc:
                best_acc = point_statistics['results'][0]
                best_drop = preopts

            statistics[model][preopts] = point_statistics

        statistics[model]['best_pre'] = best_drop

    # print results
    for model, model_stat in statistics.items():
        best_pre = model_stat['best_pre']
        print(f"Results for {model}:")
        print(f"    best average accuracy: {model_stat[best_pre]['results'][0]} ({model_stat[best_pre]['results'][1]})")
        print(f"    achieved for pre_opts: {best_pre} (dropped labels, feature cross)")
        print(f"    average parameters for preprocessing options:")
        for value, values in model_stat[best_pre].items():
            if value == 'results':
                continue
            if isinstance(values[0], str):
                print(f"        {value:>15}: {values[0]}")
                continue
            print(f"        {value:>15}: avg: {values[0]:.10f}  median: {values[2]:.10f}  std: {values[1]:.10f}")
            

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
