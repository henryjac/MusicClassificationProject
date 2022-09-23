import pandas as pd
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statistics as s
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from sklearn.svm import SVC

# ------------------- IMPORT DATA ----------------
data = pd.read_csv (r'data/project_train.csv', encoding='utf-8') # Import data
inputs = pd.DataFrame(data).to_numpy()
print(np.shape(inputs))
np.random.shuffle(inputs)
print(np.shape(inputs))

# -----------------  TRAINING DATA ---------------------------------
X_train = inputs[0:374,0:11] # training data
y_train = inputs[0:374,11]
print(X_train)
print(np.size(y_train))
# ----------------- EVALUATION DATA ---------------------------------
X_test = inputs[375:,0:11]
y_val = inputs[375:,11]
print('-----')

print(np.size(X_test))
print(np.size(y_val))


def svm(X_train, X_test, y_train):
    #SVM
    model = SVC(C=1,kernel='poly' )
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels

labels = svm(X_train, X_test, y_train)
print(labels)
acc = y_val - labels
acc = 1- np.dot(acc,acc) / len(y_val)
print(acc)