from sklearn.svm import SVC # Supprort Vector classifier
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

def svm(X_train, X_test, y_train):
    #SVM
    model = SVC(C=1,kernel='linear' )
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels

def knn(X_train, X_test, y_train):
    # initialize the model with parameter k
    KNNClassifier = KNeighborsClassifier(n_neighbors=10, weights='distance',algorithm='auto')
    # train the model with X_train datapoints and y_train data labels
    KNNClassifier.fit(X_train, y_train)
    # returns a classification of a X_test datapoints
    labels_knn = KNNClassifier.predict(X_test)
    return labels_knn

def lda(X_train, X_test, y_train):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels

def qda(X_train, X_test, y_train):
    model = QuadraticDiscriminantAnalysis( reg_param=0.01)
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels

def rfc(X_train, X_test, y_train):
    model = RandomForestClassifier(criterion='entropy', n_estimators=200, max_features='sqrt')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels
