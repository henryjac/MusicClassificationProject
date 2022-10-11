import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as clr
import pandas as pd
import numpy as np
import os

def similarity_check(label1, label2):
    df1 = pd.read_csv(label1, header=None)
    df2 = pd.read_csv(label2, header=None)

    length = df1.size
    acc = (length - (df1 - df2).abs().sum(axis=1))/length
    return acc

def cross_similarity_check(directory='labels/old/',savelocation='img/similarity_results_old.svg'):
    label_files = os.listdir(directory)

    def csvs(string):
        return "csv" in string
    label_files = list(filter(csvs, label_files))
    matrix = np.zeros((len(label_files),len(label_files)))
    labels = label_files.copy()
    label_files = [directory+label for label in label_files]

    length = 0
    for i,label1 in enumerate(label_files):
        length += 1
        for j,label2 in enumerate(label_files[:i]):
            matrix[i,j] = similarity_check(label1,label2)
            matrix[j,i] = matrix[i,j]
    matrix = pd.DataFrame(matrix, labels, labels)

    figsize = (10+length//7,10+length//7)

    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.set_style(style = 'white')
    f, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(10, 250, as_cmap=True)
    sns.heatmap(matrix, mask=mask, cmap=cmap,
                        square=True,
                        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.title('Similarity between results', fontsize=20)
    plt.savefig(savelocation)

def main():
    cross_similarity_check()
    # cross_similarity_check('./','img/similarity_results_new.svg')

if __name__ == '__main__':
    main()
