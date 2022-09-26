import numpy as np

from sklearn.svm import SVC # Supprort Vector classifier
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

def ML_model_prediction(X_train, X_test, y_train, ML_model, **kwargs):
    model = ML_model(**kwargs)
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels

def get_accuracy(X_train, X_test, y_train, y_test, ML_model, **kwargs):
    labels = ML_model_prediction(X_train, X_test, y_train, ML_model, **kwargs)
    acc = y_test - labels
    acc = 1 - np.dot(acc,acc) / len(y_test)
    return acc
