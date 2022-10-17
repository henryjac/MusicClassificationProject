import random
import sklearn

import pandas as pd
import statistics as s
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import sys,os
sys.path.append('src')
import models

def preprocessing(drop=[], keep=[], use_percentage=1, feature_crosses=[]):
    """
    Preprocesses the project_train.csv file by removing irrelevant columns
    and rows.

    :param X_test: Remove same columns of the test data as training data
    :param drop: List of column names/indices to drop. default=None
    :param use_percentage: The fractional part of the dataframe to use. defualt=1
    :return: The DataFrame object
    """
    train_file = 'data/project_train.csv'
    test_file = 'data/project_test.csv'
    X_test = pd.read_csv(test_file, encoding='utf-8')
    data = pd.read_csv(train_file, encoding='utf-8')
    data = pd.DataFrame(data)

    # Remove rows with faulty values, which
    # we found in the labels 'energy' and 'loudness'
    energy_idx_err = data.idxmax()['energy']
    loudness_idx_err = data.idxmin()['loudness']
    data = data.drop([energy_idx_err, loudness_idx_err], axis=0)

    # Use only a random percentage of the data
    data = data.sample(frac=use_percentage)

    if feature_crosses:
        data = feature_cross(data, *feature_crosses, is_test=False)
        X_test = feature_cross(X_test, *feature_crosses, is_test=True)

    if keep != []:
        if isinstance(keep[0],int):
            keep = data.columns[keep]
        if 'Label' not in keep:
            keep_data = pd.Index.append(keep,pd.Index(['Label']))
        data = data[keep_data]
        X_test = X_test[keep]
    elif drop != []:
        if isinstance(drop[0],int):
            drop = data.columns[drop]
        data = data.drop(drop, axis=1)
        if X_test is not None:
            X_test = X_test.drop(drop, axis=1)

    data = normalize(data) # Use deviation of mean to normalize
    X_test = normalize(X_test)
    return data, X_test

def normalize(dataframe):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

def feature_cross(df, *columns, is_test=False):
    columns = [df.columns[i] for i in columns]
    new_feature = '_x_'.join(columns)
    if is_test:
        x = [*df.columns,new_feature]
    else:
        x = [*df.columns[:-1],new_feature,"Label"]
    df[new_feature] = df[columns[0]]
    for column in columns[1:]:
        df[new_feature] *= df[column]

    df = df.reindex(columns=x)
    return df

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

def select_most_common_label():
    dfs = pd.DataFrame()
    nr_of_label_files =  0
    for fl in os.listdir('labels'):
        if 'csv' not in fl:
            continue
        nr_of_label_files += 1
        df = pd.read_csv(f'labels/{fl}', encoding='utf-8', header=None)
        dfs = pd.concat([dfs, df])
    df_sum = dfs.sum().transpose()
    df_sum[df_sum <= nr_of_label_files/2] = 0
    df_sum[df_sum > nr_of_label_files/2] = 1
    df_sum.to_numpy().tofile('labels/average_labels.csv',sep=',')

def main():
    # select_most_common_label()
    feature_cross(preprocessing(), "valence", "key")

if __name__ == '__main__':
    main()
