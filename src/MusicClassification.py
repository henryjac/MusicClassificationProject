import pandas as pd
from sklearn.svm import SVC # Supprort Vector classifier

import sys
sys.path.append('src')
import preprocessing
import models

def main():
    X_test = pd.read_csv('data/project_test.csv', encoding='utf-8')
    train_data, X_test = preprocessing.preprocessing(X_test)

    X_train = train_data.drop('Label',axis=1)
    y_train = train_data['Label']

    y_test = models.ML_model_prediction(
        X_train, X_test, y_train,
        SVC, C=1, kernel='linear'
    )

    y_test.tofile('data/labels.csv',sep='\n')

if __name__ == '__main__':
    main()
