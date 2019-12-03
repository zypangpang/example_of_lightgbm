import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score

data_path=Path('./data/NASA/NASATrain/')
file_path=data_path/'cm1train.csv'
df=pd.read_csv(str(file_path))
df.isnull().sum().any()
X_train=df.iloc[:,:-1]
y_train=df['l']
y_val=pd.concat([y_train.loc[lambda x:x==-1].sample(frac=0.3,random_state=1),y_train.loc[lambda x:x==1].sample(frac=0.3,random_state=1)])
y_trn=y_train.drop(y_val.index)
X_trn=X_train.loc[y_trn.index,:]
X_val=X_train.loc[y_val.index,:]
train_data=lgb.Dataset(X_trn,label=y_trn)
val_data=lgb.Dataset(X_val,label=y_val)
params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "learning_rate":0.2,
        "num_leaves":32
    }
num_round = 100
bst = lgb.train(params, train_data, num_round,val_data,early_stopping_rounds=70)
test_path=Path('./data/NASA/NASATest/')
test_file_path=test_path/'cm1test.csv'
df_test=pd.read_csv(str(test_file_path))
df_test_data=df_test.iloc[:,:-1]
df_test_label=df_test['l']
ypred=bst.predict(df_test_data)
print(roc_auc_score(df_test_label,ypred))
