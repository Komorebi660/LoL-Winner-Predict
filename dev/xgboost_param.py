import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from xgboost import XGBClassifier

import matplotlib.pyplot as plt


def load_data():
    with open('../data/train_processed.csv', 'r') as f:
        dataset = pd.read_csv(f)
    features = dataset.drop(['win'], axis=1).values
    labels = dataset['win'].values
    return features, labels


def run(key, values, idx=1):
    # Load training data
    features, labels = load_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    params_ = {'learning_rate': 0.01, 'n_estimators': 8000, \
                'max_depth': 5, 'gamma': 0.05, 'min_child_weight': 0.8, \
                'reg_alpha': 0.2, 'reg_lambda': 0.5, \
                'objective': 'binary:logistic', \
                'subsample': 0.8, 'colsample_bytree': 0.8, \
                'random_state': 42, 'n_jobs': 8}
    
    scores = []
    for value in values:
        params_[key] = value
        classifier = XGBClassifier(**params_)
        score = cross_val_score(classifier, features, labels, 
                                cv=kf, scoring='accuracy', n_jobs=5)
        scores.append(score.mean())
        print(f'{key}={value} done.')
    
    plt.subplot(3, 3, idx)
    plt.plot(values, scores, marker='o')
    plt.xlabel(key)
    plt.ylabel('Accuracy')
    plt.grid()


if __name__ == '__main__':
    plt.figure(figsize=(15, 15))
    run('learning_rate', [0.001, 0.005, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1], idx=1)
    run('n_estimators', [1000, 3000, 5000, 8000, 10000, 12000], idx=2)
    run('max_depth', [3, 4, 5, 6, 7, 8, 9], idx=3)
    run('gamma', [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8], idx=4)
    run('min_child_weight', [0.1, 0.3, 0.5, 0.8, 1, 1.5, 2], idx=5)
    run('reg_alpha', [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1], idx=6)
    run('reg_lambda', [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1], idx=7)
    run('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1], idx=8)
    run('colsample_bytree', [0.5, 0.6, 0.7, 0.8, 0.9, 1], idx=9)
    plt.tight_layout()
    plt.savefig('./xgb_parameter.png')
    