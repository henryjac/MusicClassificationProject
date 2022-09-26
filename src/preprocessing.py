import pandas as pd
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statistics as s
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from sklearn.svm import SVC # Supprort Vector classifier
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
# ------------------- IMPORT DATA ----------------
data = pd.read_csv (r'data/project_train.csv', encoding='utf-8') # Import data
inputs = pd.DataFrame(data).to_numpy()
print('size of data input')
print(np.shape(inputs))
np.random.shuffle(inputs)
# -----------------  TRAINING DATA ---------------------------------
X_train = inputs[0:374,0:11] # training data (without the labels)
y_train = inputs[0:374,11] # training data (only labels)
# ----------------- EVALUATION DATA ---------------------------------
X_test = inputs[375:,0:11] # Using last 25% of data to evaluate how good the classifier is.
y_val = inputs[375:,11] # correct labels on the evaluation data.
print('-----')

def svm(X_train, X_test, y_train):
    #SVM
    model = SVC(C=1,kernel='linear' )
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels

labels = svm(X_train, X_test, y_train)
print(labels)
acc = y_val - labels
acc = 1 - np.dot(acc,acc) / len(y_val)
print('Accuracy with SVM')
print(acc)

def knn(X_train, X_test, y_train):    
    # initialize the model with parameter k
    KNNClassifier = KNeighborsClassifier(n_neighbors=10, weights='distance',algorithm='auto')
    # train the model with X_train datapoints and y_train data labels
    KNNClassifier.fit(X_train, y_train)
    # returns a classification of a X_test datapoints
    labels_knn = KNNClassifier.predict(X_test)
    return labels_knn

labels_knn = knn(X_train, X_test, y_train)
acc_knn = y_val - labels_knn
acc_knn = 1 - np.dot(acc_knn,acc_knn) / len(y_val)
print('Accuracy with KNN')
print(acc_knn)

def lda(X_train, X_test, y_train):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels

labels_lda = lda(X_train, X_test, y_train)
acc_lda = y_val - labels_lda
acc_lda = 1 - np.dot(acc_lda,acc_lda) / len(y_val)
print('Accuracy with LDA')
print(acc_lda)

def qda(X_train, X_test, y_train):
    model = QuadraticDiscriminantAnalysis( reg_param=0.01)
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels

labels_qda = qda(X_train, X_test, y_train)
acc_qda = y_val - labels_qda
acc_qda = 1 - np.dot(acc_qda,acc_qda) / len(y_val)
print('Accuracy with QDA')
print(acc_qda)

def rfc(X_train, X_test, y_train):
    model = RandomForestClassifier(criterion='entropy', n_estimators=200, max_features='sqrt')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels

labels_rfc = rfc(X_train, X_test, y_train)
acc_rfc = y_val - labels_rfc
acc_rfc = 1 - np.dot(acc_rfc,acc_rfc) / len(y_val)
print('Accuracy with rfc')
print(acc_rfc)