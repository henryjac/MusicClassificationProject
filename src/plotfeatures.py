import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as clr

from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.append('src')
import preprocessing

def main():
    # plot_all_features()
    correlation_plot()
    feature_importance()

def get_data():
    data = pd.read_csv('data/project_train.csv', encoding='utf-8')

    # Find the indices where we had clearly faulty values
    energy_err = data.idxmax()['energy']
    loudness_err = data.idxmin()['loudness']
    data = data.drop([energy_err, loudness_err])

    return data

def correlation_plot():
    data = get_data()
    data = preprocessing.normalize(data)

    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.set_style(style = 'white')
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(10, 250, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap,
                        square=True,
                        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.title('Correlation between features', fontsize=20)
    plt.savefig('img/correlation.svg')

def plot_all_features():
    data = get_data()
    data_np = data.to_numpy()

    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[1]):
            if i == j:
                continue
            feature_plot(data_np, i, j, data.keys()[i], data.keys()[j])

def feature_plot(data, f1, f2, namef1, namef2):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cmap, nmap = clr.from_levels_and_colors([0,0.5,1],['blue','red'])
    plt.scatter(data[:,f1],data[:,f2], c=data[:,-1], cmap=cmap)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlabel(namef1)
    plt.ylabel(namef2)

    plt.tight_layout()
    plt.savefig(f'img/feature_vs_feature/{f1}_vs_{f2}.svg')
    plt.close()

def feature_importance():
    data = get_data()
    data = preprocessing.normalize(data)

    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :-1], data.iloc[:, -1], test_size=0.3
    )
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    model.fit(X_train, y_train)

    features = data.drop('Label', axis=1).columns
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(30,15))
    plt.title('Feature importances', fontsize=30)
    plt.bar([features[i] for i in indices], importances[indices], color='b', align='center')
    plt.xticks(range(len(indices)), [features[i] for i in indices], fontsize=20)
    plt.ylabel('Relative Importance', fontsize=24)
    plt.savefig('img/feature_importances.svg')

if __name__ == '__main__':
    main()
