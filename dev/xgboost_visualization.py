import pandas as pd

from xgboost import XGBClassifier
from xgboost import plot_tree, plot_importance

import matplotlib.pyplot as plt


def load_data():
    with open(f'../data/train_processed.csv', 'r') as f:
        dataset = pd.read_csv(f)
    columns = [col for col in dataset.drop(['win'], axis=1).columns]
    features = dataset.drop(['win'], axis=1).values
    labels = dataset['win'].values
    return features, labels, columns


def run():
    # Load training data
    features, labels, columns = load_data()
    params_ = {'learning_rate': 0.01, 'n_estimators': 8000, \
                'max_depth': 5, 'gamma': 0.05, 'min_child_weight': 0.8, \
                'reg_alpha': 0.2, 'reg_lambda': 0.5, \
                'objective': 'binary:logistic', \
                'subsample': 0.8, 'colsample_bytree': 0.8, \
                'random_state': 42, 'n_jobs': 16 }
    classifier = XGBClassifier(**params_)
    classifier.fit(features, labels)
    classifier.get_booster().feature_names = columns
    return classifier


def _plot_importance(classifier):
    plot_importance(classifier, max_num_features=39, 
                    xlabel='# of occurrences in trees',
                    title='', importance_type='weight',
                    values_format='{v:.0f}', xlim=(0,15000))
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    plt.tight_layout()
    plt.savefig('../figs/xgboost_feature_weight.png')
    plt.clf()

    plot_importance(classifier, max_num_features=39, 
                    xlabel='performance gain',
                    title='', importance_type='gain',
                    values_format='{v:.2f}', xlim=(0,480))
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    plt.tight_layout()
    plt.savefig('../figs/xgboost_feature_gain.png')
    plt.clf()


def _plot_tree(classifier, i):
    plot_tree(classifier, num_trees=i-1)
    fig = plt.gcf()
    fig.set_size_inches(50, 6)
    plt.tight_layout()
    plt.savefig(f'../figs/tree_{i}.png')
    plt.clf()


if __name__ == '__main__':
    classifier = run()
    _plot_importance(classifier)
    for i in [1, 10, 100, 1000, 5000, 8000]:
        _plot_tree(classifier, i)
    