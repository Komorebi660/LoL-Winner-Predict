from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import pandas as pd
import argparse

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def load_data():
    with open('../data/train_processed.csv', 'r') as f:
        dataset = pd.read_csv(f)
    features = dataset.drop(['win'], axis=1).values
    labels = dataset['win'].values
    return features, labels


def gridsearch(classifier, param_grid):
    # read data
    X_train , Y_train = load_data()

    # grid search
    grid_search = GridSearchCV(classifier, param_grid, scoring='accuracy', \
                               cv=5, n_jobs=8, verbose=1)
    grid_search.fit(X_train, Y_train)
    print(f'Best Grid parameters On Train: {grid_search.best_params_}')
    print(f'Best Grid score On Train: {grid_search.best_score_}')

    return grid_search.best_params_


def run(classifier):
    # Load training data
    features, labels = load_data()
    print(features.shape, labels.shape)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    score = cross_val_score(classifier, features, labels, 
                            cv=kf, scoring='accuracy', n_jobs=5)
    print(f'Cross validation score: {score}, {score.mean()}')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, help='algorithm name')
    args = parser.parse_args()

    if args.algo == 'svm':
        classifier = SVC(random_state=42, verbose=1)
        param_grid = {
            'C': [0.5, 0.8, 1, 2],
            'kernel': ['linear', 'rbf', 'poly'],
            'degree': [2, 3, 4],
            'gamma': [0.01, 0.05, 0.1, 0.5, 1, 2]
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'C': 2, 'kernel': 'rbf', 'random_state': 42, 'verbose': 1}
        run(SVC(**best_params_))
    elif args.algo == 'dt':
        classifier = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [5, 8, 10, 12, 15],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'criterion': 'gini', 'max_depth': 10, \
                        'max_features': 'sqrt', 'min_samples_leaf': 1, \
                        'min_samples_split': 5, 'random_state': 42}
        run(DecisionTreeClassifier(**best_params_))
    elif args.algo == 'xgb':
        classifier = XGBClassifier(objective='binary:logistic', \
                                subsample=0.8, colsample_bytree=0.8, \
                                random_state=42, n_jobs=16)
        param_grid = {
            'learning_rate': [0.005, 0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6, 7],
            'n_estimators': [1000, 2000, 5000, 8000, 10000, 12000],
            'min_child_weight': [0.5, 0.7, 0.8, 1, 1.1],
            'gamma': [0, 0.05, 0.1, 0.2, 0.5, 0.8, 1],
            'reg_alpha': [0, 0.1, 0.5, 0.8, 1],
            'reg_lambda': [0, 0.1, 0.5, 0.8, 1]
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'learning_rate': 0.01, 'n_estimators': 8000, \
                        'max_depth': 5, 'gamma': 0.8, 'min_child_weight': 0.5, \
                        'reg_alpha': 0.8, 'reg_lambda': 0.8, \
                        'objective': 'binary:logistic', \
                        'subsample': 0.6, 'colsample_bytree': 0.8, \
                        'random_state': 42, 'n_jobs': 16}
        run(XGBClassifier(**best_params_))
    elif args.algo == 'rf':
        classifier = RandomForestClassifier(random_state=42, n_jobs=16, verbose=1)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [5, 8, 10, 12, 15],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10],
            'n_estimators': [500, 1000, 1500, 2000],
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'criterion': 'gini', 'max_depth': 10, \
                       'max_features': 'sqrt', 'n_estimators': 1000, \
                       'min_samples_leaf': 5, 'min_samples_split': 10, \
                       'random_state': 42, 'n_jobs': 16}
        run(RandomForestClassifier(**best_params_))
    elif args.algo == 'nb':
        classifier = GaussianNB()
        run(classifier)
    elif args.algo == 'lr':
        classifier = LogisticRegression(random_state=42, n_jobs=16)
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.5, 0.8, 1, 2],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [1000, 2000, 3000, 4000, 5000]
        }
        #best_params_ = gridsearch(classifier, param_grid)
        best_params_ = {'C': 2, 'max_iter': 1000, 'penalty': 'l2', 'solver': 'lbfgs', \
                        'random_state': 42, 'n_jobs': 16}
        run(LogisticRegression(**best_params_))
    else:
        print('Invalid algorithm name!')