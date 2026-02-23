## package imports
import numpy as np
import pandas as pd
from pickle import dump, load
import os
from sklearn.preprocessing import OrdinalEncoder
import skops.io as sio
import seaborn as sns
import matplotlib.pyplot as plt

## This code is a pipeline of preprocessing-feature engineering-feature selection
## solely for grading purpose
## More detail on preprocessing, feature_engineering, and feature_selection notebooks

def preprocessing():
    """
    Preprocessing data from eda.ipynb result
    """
    ## dataset loading
    with open('data/eda/data.pkl', 'rb') as file:
        df = load(file)

    ## defense against missing predictor
    mp_check = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'Exited']
    cols = list(df.columns)
    not_exist = []
    for i in mp_check:
        if i not in cols:
            not_exist.append(i)

    if len(not_exist)>0:
        raise ValueError(f"Some columns are missing: {not_exist}")
    
    ## Handling against wrong data type
    wt_check = {
        'RowNumber': 'int64',
        'CustomerId': 'int64',
        'Surname': 'object',
        'CreditScore': 'int64',
        'Geography': 'object',
        'Gender': 'object',
        'Age': 'int64',
        'Tenure': 'int64',
        'Balance': 'float64',
        'EstimatedSalary': 'float64',
        'Exited': 'int64'
    }

    for i in cols:
        if df[i].dtype != wt_check[i]:
            print(f'Column {i} has wrong data type')

    ## numerical on categorical becomes NaN
    def str_check(x):
        if isinstance(x, str):
            return x
        else:
            return np.nan
        
    df['Surname'] = df['Surname'].apply(str_check)
    df['Geography'] = df['Geography'].apply(str_check)
    df['Gender'] = df['Gender'].apply(str_check)

    ## string on numerical handled by pd.to_numeric
    df['RowNumber'] = pd.to_numeric(df['RowNumber'], errors='coerce').astype('int64')
    df['CustomerId'] = pd.to_numeric(df['CustomerId'], errors='coerce').astype('int64')
    df['CreditScore'] = pd.to_numeric(df['CreditScore'], errors='coerce').astype('int64')
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').astype('int64')
    df['Tenure'] = pd.to_numeric(df['Tenure'], errors='coerce').astype('int64')
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    df['EstimatedSalary'] = pd.to_numeric(df['EstimatedSalary'], errors='coerce')
    df['Exited'] = pd.to_numeric(df['Exited'], errors='coerce').astype('int64')

    ## handling against missing data and outlier
    ## handling missing data is performed on modelling
    ## here, only outliers are removed

    ## load outlier data
    with open('data/eda/out_data.pkl', 'rb') as file:
        out_df = load(file)

    df_diff = df.merge(out_df, how='left',
                    left_index=True, right_index=True,
                    indicator=True)

    in_col = [i+'_y' for i in cols]
    in_col.append('_merge')

    in_df = df_diff[df_diff['_merge'] == 'left_only']
    in_df = in_df.drop(in_col, axis=1)

    in_col = [i+'_x' for i in cols]
    in_dict = dict(zip(in_col, cols))

    in_df = in_df.rename(in_dict, axis=1)

    ## no handling on imbalanced/skewed data thanks to model's support on stratified sampling, class_weight and tree-based ensembles

    ## saving
    if not os.path.exists('data/preprocessed'):
        os.makedirs('data/preprocessed')

    with open('data/preprocessed/data.pkl', 'wb') as file:
        dump(in_df, file)

    return None

def feature_enigneering():
    """
    perform feature engineering from preprocessing function
    """
    ## dataset loading
    with open('data/preprocessed/data.pkl', 'rb') as file:
        train_data = load(file)

    with open('data/raw/test_data.pkl', 'rb') as file:
        test_data = load(file)

    ## separate feature and target
    X_train = train_data.copy().drop('Exited', axis=1)
    y_train = train_data['Exited'].copy()
    X_train = X_train.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    X_test = test_data.copy().drop('Exited', axis=1)
    y_test = test_data['Exited'].copy()
    X_test = X_test.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    ## train-test split already done in download_data.py

    ## label encoding
    ## fitting...
    geo_class = ['France', 'Germany', 'Spain']
    g_class = ['Female', 'Male']

    X_ord = X_train[['Geography', 'Gender']]
    enc_cat = [geo_class, g_class]

    enc = OrdinalEncoder(categories=enc_cat, ## enable pre-defined categories
                        handle_unknown='use_encoded_value', ## if a category was not learned at fitting...
                        unknown_value=np.nan) ## assume NaN

    enc.fit(X_ord)

    ## transform...
    def enc_transform(X, enc):
        cat_feat = enc.get_feature_names_out()
        X_ord = X[cat_feat]
        X_enc = enc.transform(X_ord)
        X_enc = pd.DataFrame(X_enc, index=X.index,
                            columns=enc.get_feature_names_out())
        
        X[cat_feat] = X_enc
        return X

    X_train = enc_transform(X_train, enc)
    X_test = enc_transform(X_test, enc)

    ## saving
    if not os.path.exists('data/feature_eng'):
        os.makedirs('data/feature_eng')

    with open('data/feature_eng/X_train.pkl', 'wb') as file:
        dump(X_train, file)

    with open('data/feature_eng/y_train.pkl', 'wb') as file:
        dump(y_train, file)

    with open('data/feature_eng/X_test.pkl', 'wb') as file:
        dump(X_test, file)

    with open('data/feature_eng/y_test.pkl', 'wb') as file:
        dump(y_test, file)

    obj_skops = sio.dump(enc, "data/feature_eng/encoder.skops")

    return None

def feature_selection():
    """
    Feature selection from input from feature_engineering function
    """
    ## dataset loading
    with open('data/feature_eng/X_train.pkl', 'rb') as file:
        X_train = load(file)

    with open('data/feature_eng/y_train.pkl', 'rb') as file:
        y_train = load(file)

    with open('data/feature_eng/X_test.pkl', 'rb') as file:
        X_test = load(file)

    with open('data/feature_eng/y_test.pkl', 'rb') as file:
        y_test = load(file)

    ## draw correlation heatmap
    X_num = X_train[['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']]

    corr_matrix = X_num.corr(method='spearman')

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")

    plt.show()

    ## remove columns
    ## no columns needs to be removed

    ## saving
    if not os.path.exists('data/feature_sel'):
        os.makedirs('data/feature_sel')

    with open('data/feature_sel/X_train.pkl', 'wb') as file:
        dump(X_train, file)

    with open('data/feature_sel/y_train.pkl', 'wb') as file:
        dump(y_train, file)

    with open('data/feature_sel/X_test.pkl', 'wb') as file:
        dump(X_test, file)

    with open('data/feature_sel/y_test.pkl', 'wb') as file:
        dump(y_test, file)

    return None

## main code
preprocessing()
feature_enigneering()
feature_selection()