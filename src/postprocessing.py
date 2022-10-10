import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as clr
import pandas as pd
import numpy as np
import os

def similarity_check(label1, label2):
    df1 = pd.read_csv(label1, header=None)
    df2 = pd.read_csv(label2, header=None)

    tot = df1.size
    sum = 0
    for i in range(tot):
        if df1[i][0] == df2[i][0]:
            sum+=1
    acc = sum/tot
    return acc

def main():
    label1 = "labels/tested/rfc_nest100_drop248_labels.csv"

    label_files = os.listdir("labels")

    def csvs(string):
        return "csv" in string
    label_files = list(filter(csvs, label_files)) 
    matrix = np.zeros((len(label_files),len(label_files)))
    labels = label_files.copy()
    label_files = ["labels/"+label for label in label_files]

    length = 0
    for i,label1 in enumerate(label_files):
        length += 1
        for j,label2 in enumerate(label_files[:i]):
            matrix[i,j] = similarity_check(label1,label2)
            matrix[j,i] = matrix[i,j]
    matrix = pd.DataFrame(matrix, labels, labels)

    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.set_style(style = 'white')
    f, ax = plt.subplots(figsize=(20, 19))
    cmap = sns.diverging_palette(10, 250, as_cmap=True)
    sns.heatmap(matrix, mask=mask, cmap=cmap,
                        square=True,
                        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.title('Similarity between results', fontsize=20)
    plt.savefig('img/similarity_results.svg')

if __name__ == '__main__':
    main()
