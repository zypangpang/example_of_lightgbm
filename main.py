import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score
import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
import utils

def train_val_split(X_train,y_train):
    y_val = pd.concat([y_train.loc[lambda x: x == -1].sample(frac=0.3, random_state=1),
                       y_train.loc[lambda x: x == 1].sample(frac=0.3, random_state=1)])
    y_trn = y_train.drop(y_val.index)
    X_trn = X_train.loc[y_trn.index, :]
    X_val = X_train.loc[y_val.index, :]
    return X_trn,y_trn,X_val,y_val

def train_lightgbm(params, X_trn, y_trn, X_val,y_val):#, test_data, test_label):
    train_data = lgb.Dataset(X_trn, label=y_trn)
    val_data = lgb.Dataset(X_val, label=y_val)

    num_round = 500
    model = lgb.train(params, train_data, num_round, val_data, early_stopping_rounds=100)
    return model
    #ypred = bst.predict(test_data)
    #print(roc_auc_score(test_label, ypred))


def hyperopt_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params):
    ## fixed lightgbm params
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
        #"learning_rate": 0.12524,
        #"num_leaves": 16
    }

    ## space for var_threshold search
    space1 = hp.choice('var_threshold',np.linspace(5,20,15,dtype=int))

    ## space for lightgbm hyperparam search
    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
        # "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(16, 64, 4, dtype=int)),
        # "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        # "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        # "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 10, 1, dtype=int)),
        # "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        # "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        # "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
        # "scale_pos_weight": hp.uniform('x', 0, 5),
    }

    var_series=X_train.var()

    def objective(hyperparams):
        #model = lgb.train({**params, **hyperparams}, train_data, 500,
        #                  valid_data, early_stopping_rounds=100, verbose_eval=0)
        #X_trn=X_train.loc[:,var_series>hyperparams]
        X_trn=X_train
        X_trn,y_trn,X_val,y_val=train_val_split(X_trn,y_train)

        model=train_lightgbm({**params,**hyperparams},X_trn,y_trn,X_val,y_val)
        #model=train_lightgbm(params,X_trn,y_trn,X_val,y_val)

        score = model.best_score["valid_0"]["auc"]

        feature_importance_df = pd.DataFrame()
        feature_importance_df["features"] = X_trn.columns
        feature_importance_df["importance_gain"] = model.feature_importance(importance_type='gain')
        record_zero_importance = feature_importance_df[feature_importance_df["importance_gain"] == 0.0]
        to_drop = list(record_zero_importance['features'])

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK, "drop_feature": to_drop, "best_iter": model.best_iteration}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=100, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    print(f"hyperopt auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    drop_feature = set(X_train.columns.tolist())
    for result in trials.results:
        drop_feature = drop_feature & set(result['drop_feature'])
    print(drop_feature)
    return hyperparams, drop_feature, trials.best_trial['result']['best_iter']

@utils.debug_wrapper
def main():
    data_path = Path('./data/NASA/NASATrain/')
    file_path = data_path / 'cm1train.csv'
    df = pd.read_csv(str(file_path))
    df.isnull().sum().any()
    X_train = df.iloc[:, :-1]
    y_train = df['l']
    hyperopt_lightgbm(X_train,y_train,{})

main()
