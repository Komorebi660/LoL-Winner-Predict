import numpy as np
import pandas as pd

from gplearn.genetic import SymbolicTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# 利用决策树获得最优分箱的边界值列表
def optimal_binning_boundary(x, y, max_bins, min_x, max_x):
    x = x.values  
    y = y.values
    
    clf = DecisionTreeClassifier(min_samples_split=0.1,
                                 min_samples_leaf=0.05,
                                 max_leaf_nodes=max_bins,
                                 random_state=42)

    clf.fit(x.reshape(-1, 1), y)  # 训练决策树
    
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    
    boundary = [] 
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])

    boundary.sort()
    boundary = [min_x] + boundary + [max_x]
    return boundary


def load_data():
    full_df = {}
    for split in ['train', 'test']:
        with open(f'./data/{split}.csv', 'r') as f:
            dataset = pd.read_csv(f).drop(['id', 'timecc'], axis=1)

        if split == 'test':
            dataset['win'] = np.zeros(dataset.shape[0]).astype(np.int64)

        # log transform
        for col_name in dataset.columns:
            if col_name != 'win':
                dataset[col_name] = np.log1p(dataset[col_name])

        full_df[split] = dataset

    return full_df


def split_into_bins(full_df):
    # merge train, test
    all_data = pd.concat([full_df['train'], full_df['test']], axis=0, ignore_index=True)

    # split into bins
    col_name = ['kills', 'deaths', 'assists', 'largestkillingspree', 'largestmultikill', 
                'longesttimespentliving', 'doublekills', 'triplekills', 'quadrakills', 
                'pentakills', 'totdmgdealt', 'magicdmgdealt', 'physicaldmgdealt', 'truedmgdealt', 
                'largestcrit', 'totdmgtochamp', 'magicdmgtochamp', 'physdmgtochamp', 
                'truedmgtochamp', 'totheal', 'totunitshealed', 'dmgtoturrets', 'totdmgtaken', 
                'magicdmgtaken', 'physdmgtaken', 'truedmgtaken', 'wardsplaced', 'wardskilled']
    for col in col_name:
        min_value = all_data[col].min()-0.1
        max_value = all_data[col].max()+0.1
        bins = optimal_binning_boundary(full_df['train'][col], full_df['train']['win'], 100, min_value, max_value)
        #print(col, bins)
        for split in ['train', 'test']:
            full_df[split][col] = pd.cut(full_df[split][col], bins=bins, labels=[i for i in range(len(bins)-1)], right=True).astype(np.int64)
    
    return full_df


def feature_mining(full_df):
    scaler = StandardScaler()
    st = SymbolicTransformer(
        generations=20,
        population_size=2000,
        hall_of_fame=100,
        n_components=10,
        function_set=['add', 'sub', 'mul', 'div'],
        parsimony_coefficient=0.0005,
        max_samples=0.8,
        init_depth=(2, 4),
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.1,
        p_point_mutation=0.1,
        p_point_replace=0.1,
        metric='spearman',
        verbose=1,
        random_state=42,
        n_jobs=16
    )
    model = RandomForestClassifier(random_state=42, n_jobs=16, verbose=1)
    rfe = RFE(model, n_features_to_select=36, verbose=1)

    X_train = full_df['train'].drop(['win'], axis=1).values.copy().astype(np.float64)
    Y_train = full_df['train']['win'].values.copy().astype(np.int32)
    print(X_train.shape, Y_train.shape)
    # normalize
    X_train = scaler.fit_transform(X_train)
    # mine features by genetic programming
    X_mined_train = st.fit_transform(X_train, Y_train)
    print(X_mined_train.shape)
    X_train = np.concatenate((X_mined_train, X_train), axis=1)
    print(X_train.shape)
    '''# remove unimportant features
    X_train = rfe.fit_transform(X_train, Y_train) 
    print(X_train.shape)'''

    for split in ['train', 'test']:
        _features = full_df[split].drop(['win'], axis=1)
        labels = full_df[split]['win']

        features_value = scaler.transform(_features.values.astype(np.float64))
        features = pd.DataFrame(features_value, columns=_features.columns)

        mined_features_value = st.transform(features_value)
        mined_features = pd.DataFrame(mined_features_value, 
                                      columns=[f'mined_feature_{i}' for i in range(mined_features_value.shape[1])])
        
        features = pd.concat([mined_features, features], axis=1)
        #features = features[features.columns[rfe.support_]]
        print(features.shape)

        # save new data
        final_features = pd.concat([features, labels], axis=1)
        final_features.to_csv(f'./data/{split}_processed.csv', index=False)


if __name__ == '__main__':
    print("Start data preprocess...")
    full_df = load_data()
    #print("Split into bins...")
    #full_df = split_into_bins(full_df)
    print("Feature mining...")
    feature_mining(full_df)