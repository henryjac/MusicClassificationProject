import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def main():
    # plot_all_features()
    correlation_plot()

def get_data(numpy = True):
    data = pd.read_csv('data/project_train.csv', encoding='utf-8')
    inputs = pd.DataFrame(data)

    # Find the indices where we had clearly faulty values
    energy_err = inputs.idxmax()['energy']
    loudness_err = inputs.idxmin()['loudness']
    inputs = inputs.drop([energy_err, loudness_err])

    if numpy:
        return data, inputs.to_numpy()
    return data, inputs

def correlation_plot():
    _, inputs = get_data(False)

    corr = inputs.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.set_style(style = 'white')
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(10, 250, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap,
                        square=True,
                        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.title('Correlation between features')
    plt.savefig('img/correlation.svg')

def plot_all_features():
    data, inputs = get_data()

    for i in range(np.shape(inputs)[1]):
        for j in range(np.shape(inputs)[1]):
            if i == j:
                continue
            feature_plot(inputs, i, j, data.keys()[i], data.keys()[j])

def feature_plot(inputs, f1, f2, namef1, namef2):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cmap, nmap = clr.from_levels_and_colors([0,0.5,1],['blue','red'])
    plt.scatter(inputs[:,f1],inputs[:,f2], c=inputs[:,-1], cmap=cmap)
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlabel(namef1)
    plt.ylabel(namef2)

    plt.tight_layout()
    plt.savefig(f'img/feature_vs_feature/{f1}_vs_{f2}.svg')
    plt.close()

if __name__ == '__main__':
    main()
