import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()


def draw_hist(data):
    fig, axes = plt.subplots(10, 4, figsize=(20, 40))
    for idx, col in enumerate(data.columns):
        if col == 'win':
            continue
        sns.histplot(data=data, x=col, ax=axes[idx//4, idx%4], 
                    kde=True, stat="probability", hue='win',
                    element="bars", common_norm=False, bins=50)
    plt.tight_layout()
    plt.savefig('figs/distribution.png')
    plt.clf()


def draw_box(data):
    fig, axes = plt.subplots(10, 4, figsize=(20, 40))
    for idx, col in enumerate(data.columns):
        if col == 'label':
            continue
        sns.boxplot(y=col, data=data, hue='win', ax=axes[idx//4, idx%4],
                    showfliers=True, width=.5, gap=.2)
    plt.tight_layout()
    plt.savefig('figs/difference.png')
    plt.clf()


def draw_corr(data):
    sns.set_context({"figure.figsize":(20,20)})
    sns.heatmap(data=data.corr(), square=True, cmap='RdBu_r')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('figs/corr.png')
    plt.clf()


def draw_neighbor(data):
    labels = data['win'].values
    data = data.drop(['win'], axis=1).values

    # normalize
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # random sample
    np.random.seed(0)
    idx = np.random.choice(len(labels), size=2000, replace=False)
    data = data[idx]
    labels = labels[idx]
    print(np.sum(labels))

    # t-sne
    tsne = TSNE(n_components=2, learning_rate=500, n_iter=10000, \
                metric='l1', random_state=0)
    data = tsne.fit_transform(data) 
    print(data.shape)

    # plot
    cmap = plt.cm.Spectral
    plt.figure(figsize=(10, 10))
    for i in range(2):
        indices = labels == i
        plt.scatter(data[indices, 0], data[indices, 1], color=cmap(i/1.1), label=i)
    plt.legend()
    plt.savefig('figs/neighbor.png')
    plt.clf()



if __name__ == "__main__":
    x_train = pd.read_csv('data/train_processed.csv')
    draw_hist(x_train)
    draw_box(x_train)
    draw_corr(x_train)
    draw_neighbor(x_train)
