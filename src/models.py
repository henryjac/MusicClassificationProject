import numpy as np

from sklearn.model_selection import GridSearchCV

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

# Uses 5-fold cross validation to perform gridsearch over the given
# hyperparameter space for the given model.
#
# Note, if memory issues are encountered when running this job in parallel,
# try setting pre_dispatch=2*n_jobs to reduce the memory load.
#
# returns model: the estimator function for the best model
#         param: the parameters for the best model found
def grid_search(X_train, y_train, ML_model, parameter_space, verbose=True, n_cores=28):
    search_object = GridSearchCV(ML_model(), parameter_space, n_jobs=n_cores)
    n_cases = 1
    for param in parameter_space:
        n_cases *= len(parameter_space[param])
    
    if verbose:
        print(f"Performing grid search for {ML_model.__name__} on {n_cores} cores ({n_cases} cases)...")
    search_object.fit(X_train, y_train)

    best_score = search_object.best_score_
    model = search_object.best_estimator_
    param = search_object.best_params_
    if verbose:
        print(f"    best score: {best_score}")
        print(f"    with param: {param}")
        
    return (model, param)

def cross_validate(X_train, y_train, folds, ML_model, **kwargs):
    # determine number of elements per fold
    fshapes = np.ones(folds)*np.floor(np.shape(y_train)[0]/folds)
    fshapes[0:(np.shape(y_train)[0]%folds)] += 1.0
    # fsteps[i] contains starting index of i'th fold
    fsteps = np.insert(np.cumsum(fshapes), 0, 0.0)

    # determine accuracy for each fold
    cumacc = 0.0
    for i in range(folds):
        # index vector for training set
        I_k = [*range(0, int(fsteps[i])), *range(int(fsteps[i+1]), int(fsteps[-1]))]

        xtrain_k = X_train[I_k, :]
        ytrain_k = y_train[I_k]
        xtest_k = X_train[int(fsteps[i]):int(fsteps[i+1]), :]
        ytest_k = y_train[int(fsteps[i]):int(fsteps[i+1])]

        cumacc += get_accuracy(xtrain_k, xtest_k, ytrain_k, ytest_k, ML_model, **kwargs)

    return cumacc/folds
