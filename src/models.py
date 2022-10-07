import numpy as np

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

def cross_validate(X_train, y_train, folds, ML_model, **kwargs):
    # determine number of elements per fold
    fshapes = np.ones(folds)*np.floor(np.shape(y_train)[0]/folds)
    fshapes[0:(np.shape(y_train)[0]%folds)] += 1.0
    # fsteps[i] contains starting index of i'th fold
    fsteps = np.insert(np.cumsum(fshapes), 0, 0.0)

    # determine accuracy for each fold
    accs = []
    for i in range(folds):
        # index vector for training set
        I_k = [*range(0, int(fsteps[i])), *range(int(fsteps[i+1]), int(fsteps[-1]))]

        xtrain_k = X_train[I_k, :]
        ytrain_k = y_train[I_k]
        xtest_k = X_train[int(fsteps[i]):int(fsteps[i+1]), :]
        ytest_k = y_train[int(fsteps[i]):int(fsteps[i+1])]

        accs.append(get_accuracy(xtrain_k, xtest_k, ytrain_k, ytest_k, ML_model, **kwargs))

    return (np.array(accs).mean(), np.array(accs).std())
