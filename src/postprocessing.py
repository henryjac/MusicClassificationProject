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

def cross_similarity_check(places=['labels/old/'],savelocation='img/similarity_results_old.svg'):
    def csvs(string):
        return "csv" in string
    label_files = []
    for place in places:
        if place[-1] == '/':
            to_labels = os.listdir(place)
            to_labels = list(filter(csvs, to_labels))
            to_labels = [place+label for label in to_labels]
            label_files.extend(to_labels)
        else:
            label_files.append(place)

    matrix = np.zeros((len(label_files),len(label_files)))
    labels = label_files.copy()

    length = 0
    for i,label1 in enumerate(label_files):
        length += 1
        for j,label2 in enumerate(label_files[:i]):
            matrix[i,j] = similarity_check(label1,label2)
            matrix[j,i] = matrix[i,j]
    matrix = pd.DataFrame(matrix, labels, labels)

    figsize = (10+length//5,10+length//5)

    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.set_style(style = 'white')
    f, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(10, 250, as_cmap=True)
    sns.heatmap(matrix, mask=mask, cmap=cmap,
                        square=True,
                        linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.title('Similarity between results', fontsize=20)
    plt.savefig(savelocation)

def get_model_label_dirs():
    label_dir = os.listdir('labels')
    label_dir = ['labels/'+label for label in label_dir]
    model_label_dirs = []
    for obj in label_dir:
        if 'old' in obj or 'tested' in obj:
            continue
        if os.path.isdir(obj):
            model_label_dirs.append(obj)
    return [label_dir+'/' for label_dir in model_label_dirs]

def get_nopreprocessing_labels():
    label_dirs = get_model_label_dirs()
    labels = []
    for label_dir in label_dirs:
        for label in os.listdir(label_dir):
            if 'drop' not in label:
                to_append = label_dir+label
                labels.append(to_append)
    return labels

def main():
    cross_similarity_check(get_model_label_dirs(),'img/similarity_results.svg')
    cross_similarity_check(get_nopreprocessing_labels(),'img/similarity_results_nodrop.svg')

if __name__ == '__main__':
    main()
