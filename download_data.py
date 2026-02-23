"""
Download data from kaggle, then remove duplicates and split
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pickle import dump, load
import os
from sklearn.model_selection import train_test_split

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    'willianoliveiragibin/customer-churn',
    'Customer Churn new.csv'
)

raw_df = df.copy() ## save raw data

df = df.drop_duplicates() ## drop duplicates

## data splitting
X = df.copy().drop('Exited', axis=1)
y = df['Exited'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42, ## for replication
                                                    shuffle=True, ## shuffle before split
                                                    stratify=y ## keep ratio
                                                    ## default setting is 75% training, 25% non training
                                                    )

## concat back
df_train = pd.concat([X_train, y_train], axis=1)
df_test  = pd.concat([X_test, y_test], axis=1) # type: ignore -> y_test count as Undefined but code can run

print(df_train.head())
print(df_test.head())

## saving data
if not os.path.exists('data/raw'):
    os.makedirs('data/raw')

## save raw data
with open('data/raw/raw_data.pkl', 'wb') as file:
    dump(raw_df, file)

## save training data
with open('data/raw/train_data.pkl', 'wb') as file:
    dump(df_train, file)

## save test data
with open('data/raw/test_data.pkl', 'wb') as file:
    dump(df_test, file)
