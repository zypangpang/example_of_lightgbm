import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from functools import reduce
from sklearn.metrics import roc_auc_score
import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
import utils, feature_selector

global_params = {
    'var_threshold': 15,
    'na_threshold':0.5,
    'lgb_num_round': 500,
    'lgb_early_stop_rounds': 100,
}


def train_val_split(X_train, y_train):
    y_val = pd.concat([y_train.loc[lambda x: x == -1].sample(frac=0.3, random_state=1),
                       y_train.loc[lambda x: x == 1].sample(frac=0.3, random_state=1)])
    y_trn = y_train.drop(y_val.index)
    X_trn = X_train.loc[y_trn.index, :]
    X_val = X_train.loc[y_val.index, :]
    return X_trn, y_trn, X_val, y_val


def train_lightgbm(params, X_trn, y_trn, X_val, y_val):  # , test_data, test_label):
    train_data = lgb.Dataset(X_trn, label=y_trn)
    val_data = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(params, train_data, global_params['lgb_num_round'], val_data,
                      early_stopping_rounds=global_params['lgb_early_stop_rounds'],
                      verbose_eval=100)
    return model


def hyperopt_lightgbm(X_train: pd.DataFrame, y_train: pd.Series):
    ## fixed lightgbm params
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4,
        "feature_fraction": .6,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        # "learning_rate": 0.12524,
        # "num_leaves": 16
    }

    ## space for var_threshold search
    space1 = hp.choice('var_threshold', np.linspace(5, 20, 15, dtype=int))

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

    # var_series=X_train.var()

    def objective(hyperparams):
        # X_trn=X_train.loc[:,var_series>hyperparams]
        X_trn = X_train
        X_trn, y_trn, X_val, y_val = train_val_split(X_trn, y_train)

        model = train_lightgbm({**params, **hyperparams}, X_trn, y_trn, X_val, y_val)
        # model=train_lightgbm(params,X_trn,y_trn,X_val,y_val)

        score = model.best_score["valid_0"][params['metric']]

        to_drop = X_trn.columns[model.feature_importance('gain') == 0]

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK, "drop_feature": to_drop, "best_iter": model.best_iteration}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=100, verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    print(f"hyperopt auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    # drop_feature = set(X_train.columns.tolist())
    # for result in trials.results:
    #    drop_feature = drop_feature & set(result['drop_feature'])
    drop_feature = \
    reduce(lambda r1, r2: {'drop_feature': r1['drop_feature'].union(r2['drop_feature'])}, trials.results)[
        'drop_feature']
    print(drop_feature)
    return {**params, **hyperparams}, drop_feature, trials.best_trial['result']['best_iter']


def train_for_predict(hyperparams, best_num_round, X_train, y_train):
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(hyperparams, train_data, best_num_round, verbose_eval=100)
    return model


def lightgbm_predict(model, X_test, y_test):
    ypred = model.predict(X_test)
    return roc_auc_score(y_test, ypred)


def test_dataset(train_file_path, test_file_path):
    df = pd.read_csv(str(train_file_path))
    # df.isnull().sum().any()
    X_train = df.drop(columns=['l'])
    y_train = df['l']
    y_trn = y_train

    X_trn = feature_selector.remove_many_na_col(X_train,global_params['na_threshold'])
    X_trn = feature_selector.remove_small_variance(X_trn, global_params['var_threshold'])
    hyperparams, drop_features, best_num_round = hyperopt_lightgbm(X_trn, y_trn)
    X_trn = X_trn.drop(columns=drop_features)
    print('X_trn columns:'+X_trn.columns)
    final_model = train_for_predict(hyperparams, best_num_round, X_trn, y_trn)

    df_test = pd.read_csv(str(test_file_path))
    df_test_data = df_test.drop(columns=['l'])
    df_test_label = df_test['l']

    return lightgbm_predict(final_model, df_test_data.loc[:, X_trn.columns], df_test_label)


@utils.debug_wrapper
def main():
    train_path = Path('./data/NASA/NASATrain/')
    test_path = Path('./data/NASA/NASATest/')
    aucs = {}
    for train_file_path in train_path.iterdir():
        test_file_path = test_path / train_file_path.name.replace('train', 'test')
        aucs[train_file_path.name] = test_dataset(train_file_path, test_file_path)
    print(aucs)
    utils.write_to_file(aucs, Path('./results/naive_results.json'))

