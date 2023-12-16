from sklearn.model_selection import cross_validate, StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd
import numpy as np


def load_data(split):
    with open(f'./data/{split}_processed.csv', 'r') as f:
        dataset = pd.read_csv(f)

    features = dataset.drop(['win'], axis=1).values.astype(np.float32)
    labels = dataset['win'].values.astype(np.int32)
   
    return features, labels


def predict():
    # read data
    X_train , Y_train = load_data('train')
    # Train the model
    classifier = XGBClassifier(objective='binary:logistic', \
                                learning_rate=0.01, max_depth=5, n_estimators=8000, \
                                subsample=0.8, colsample_bytree=0.8, \
                                min_child_weight=0.8, gamma=0.05, \
                                reg_alpha=0.2, reg_lambda=0.5, \
                                random_state=42, n_jobs=16)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    Y_train_pred = cross_validate(classifier, X_train, Y_train, cv=kf, 
                                  n_jobs=5, verbose=1, return_estimator=True)
    
    print('On Train:')
    print(Y_train_pred['test_score'])

    X_test, _ = load_data('test')
    Y_test_pred_score = np.zeros(X_test.shape[0])
    for idx, clf in enumerate(Y_train_pred['estimator']):
        Y_test_pred_score += clf.predict_proba(X_test)[:, 1]
    
    Y_test_pred = [1 if x > 2.4 else 0 for x in Y_test_pred_score]
    results = {'win': Y_test_pred}
    df = pd.DataFrame(results)
    df.to_csv('submission.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    predict()
    