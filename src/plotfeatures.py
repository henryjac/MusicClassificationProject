import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def main():
    data = pd.read_csv('data/project_train.csv', encoding='utf-8')
    inputs = pd.DataFrame(data).to_numpy()

    energy_err  = np.argmax(inputs[:,1])
    inputs = np.delete(inputs,energy_err,0)

    loudness_err = np.argmin(inputs[:,3])
    inputs = np.delete(inputs,loudness_err,0)

    for i in range(np.shape(inputs)[1]-1):
        for j in range(np.shape(inputs)[1]-1):
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
    plt.savefig(f'img/{f1}_vs_{f2}.svg')
    plt.close()

if __name__ == '__main__':
    main()
