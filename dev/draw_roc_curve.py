from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt


def load_data():
    with open('../data/train_processed.csv', 'r') as f:
        dataset = pd.read_csv(f)
    features = dataset.drop(['win'], axis=1).values
    labels = dataset['win'].values
    return features, labels


def run(classifier):
    # Load training data
    features, labels = load_data()
    print(features.shape, labels.shape)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    y_probas = cross_val_predict(classifier, features, labels, cv=kf, 
                                 method='predict_proba', n_jobs=16, verbose=1)
    
    # 初始化 ROC 曲线的平均值
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)

    kf_splits = list(kf.split(features, labels))

    # 计算每一折的 ROC 曲线，然后将它们加到平均值中
    for train, test in kf_splits:
        fpr, tpr, _ = roc_curve(labels[test], y_probas[test][:, 1])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

    # 计算平均值
    mean_tpr /= len(kf_splits)
    
    return mean_fpr, mean_tpr


def get_data():
    data = {}
    _best_params_ = {'criterion': 'gini', 'max_depth': 10, \
                    'max_features': 'sqrt', 'min_samples_leaf': 1, \
                    'min_samples_split': 5, 'random_state': 42}
    fpr, tpr = run(DecisionTreeClassifier(**_best_params_))
    data['decision tree'] = (fpr, tpr)

    best_params_ = {'learning_rate': 0.01, 'n_estimators': 8000, \
                    'max_depth': 5, 'gamma': 0.8, 'min_child_weight': 0.5, \
                    'reg_alpha': 0.8, 'reg_lambda': 0.8, \
                    'objective': 'binary:logistic', \
                    'subsample': 0.6, 'colsample_bytree': 0.8, \
                    'random_state': 42, 'n_jobs': 16}
    fpr, tpr = run(XGBClassifier(**best_params_))
    data['xgboost'] = (fpr, tpr)

    best_param_ = {'criterion': 'gini', 'max_depth': 10, \
                    'max_features': 'sqrt', 'n_estimators': 1000, \
                    'min_samples_leaf': 5, 'min_samples_split': 10, \
                    'random_state': 42, 'n_jobs': 16}
    fpr, tpr = run(RandomForestClassifier(**best_param_))
    data['random forest'] = (fpr, tpr)

    fpr, tpr = run(GaussianNB())
    data['naive bayes'] = (fpr, tpr)

    best_params_ = {'C': 2, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'lbfgs', \
                    'random_state': 42, 'n_jobs': 16}
    fpr, tpr = run(LogisticRegression(**best_params_))
    data['logistic regression'] = (fpr, tpr)

    # too slow
    #_best_params_ = {'C': 2, 'kernel': 'rbf', 'verbose': 1, \
    #                 'random_state': 42, 'probability': True}
    #fpr, tpr = run(SVC(**_best_params_))
    #data['svm'] = (fpr, tpr)

    return data


def draw_roc_curve():
    data = get_data()

    for key in data:
        fpr, tpr = data[key]
        plt.plot(fpr, tpr, label=key)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../figs/roc_curve.png')
    plt.clf()

    for key in data:
        fpr, tpr = data[key]
        plt.plot(fpr, tpr, label=key)
    plt.xlim([0.1, 0.4])
    plt.ylim([0.6, 0.9])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../figs/roc_curve_upper_left.png')
    plt.clf()


if __name__ == '__main__':
    draw_roc_curve()
