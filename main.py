import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from functools import reduce
from sklearn.metrics import roc_auc_score
import hyperopt
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
import utils, feature_selector, preprocess

global_params = {
    'CK_var_threshold': 1,
    'NASA_var_threshold': 30,
    'na_threshold': 0.5,
    'col_threshold': 0.98,
    'remove_collinear_threshold': 700,
    'lgb_num_round': 200,
    'lgb_early_stop_rounds': 100,
    'hyperopt_rounds': 50,
    'model_num': 15,
    'hyperopt_per_mode': False,
    'remove_non_label_cols': True,
    'last_model_weight': 8,
    'global_metric':'auc'
}


def train_val_split(X_train, y_train, random_seed):
    y_val = pd.concat([y_train.loc[lambda x: x == -1].sample(frac=0.3, random_state=random_seed),
                       y_train.loc[lambda x: x == 1].sample(frac=0.3, random_state=random_seed)])
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


def hyperopt_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val, y_val):
    ## fixed lightgbm params
    params = {
        "objective": "binary",
        "metric": global_params['global_metric'],
        "verbosity": -1,
        "seed": 1,
        "num_threads": 4,
        "feature_fraction": .6,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        # "learning_rate": 0.1,
        # "num_leaves": 32
    }

    ## space for var_threshold search
    space1 = hp.choice('var_threshold', np.linspace(0, 20, 15, dtype=int))

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

    var_series = X_train.var()

    def objective(hyperparams):
        # X_trn=X_train.loc[:,var_series>hyperparams]
        # X_trn = X_train
        # X_trn, y_trn, X_val, y_val = train_val_split(X_trn, y_train,random_seed)

        model = train_lightgbm({**params, **hyperparams}, X_train, y_train, X_val, y_val)
        # model=train_lightgbm(params,X_trn,y_trn,X_val,y_val)

        score = model.best_score["valid_0"][global_params['global_metric']]

        to_drop = X_train.columns[model.feature_importance('gain') == 0]
        print(f'to drop:{len(to_drop)}')

        # in classification, less is better
        return {'loss': -score, 'status': STATUS_OK, "drop_feature": to_drop, "best_iter": model.best_iteration}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                         algo=tpe.suggest, max_evals=global_params['hyperopt_rounds'], verbose=1,
                         rstate=np.random.RandomState(1))

    hyperparams = space_eval(space, best)
    print(f"hyperopt auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

    drop_feature = \
        reduce(lambda r1, r2: {'drop_feature': r1['drop_feature'].union(r2['drop_feature'])}, trials.results)[
            'drop_feature']
    print(f'drop features:{len(drop_feature)}')
    return {**params, **hyperparams}, drop_feature, trials.best_trial['result']['best_iter']


def train_all_data(hyperparams, best_num_round, X_train, y_train):
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(hyperparams, train_data, best_num_round, verbose_eval=100)
    return model


def lightgbm_predict(models, X_test, y_test):
    res_dict = {}
    for index, model in enumerate(models):
        ypred = model.predict(X_test)
        res_dict[f'model_{index}'] = ypred
        print(f'model_{index} predict finished')

    res_df = pd.DataFrame(res_dict)
    res_df.iloc[:, -1] = res_df.iloc[:, -1]*global_params['last_model_weight']

    return roc_auc_score(y_test, res_df.mean(axis=1))


def test_dataset(dataset_name, train_file_path, test_file_path):
    df = pd.read_csv(str(train_file_path))
    df = preprocess.process_extra_label(df, global_params['remove_non_label_cols'])
    print('finish reading train data')

    df_test = pd.read_csv(str(test_file_path))
    df_test = preprocess.process_extra_label(df_test, global_params['remove_non_label_cols'])
    print('finish reading test data')

    df_test_data = df_test.drop(columns=['l'])
    df_test_label = df_test['l']

    # df.isnull().sum().any()
    X_train = df.drop(columns=['l'])
    y_train = df['l']
    y_trn = y_train

    X_trn = feature_selector.remove_many_na_col(X_train, global_params['na_threshold'])
    print('finish remove na cols')

    X_trn = feature_selector.remove_single_unique(X_trn)
    print('finish remove single unique cols')

    X_trn = feature_selector.remove_small_variance(X_trn, global_params[f'{dataset_name}_var_threshold'])
    print('finish remove small var cols')

    if len(X_trn.index) < global_params['remove_collinear_threshold']:
        X_trn = feature_selector.remove_collinear_col(X_trn, global_params['col_threshold'])
        print('finish remove collinear cols')

    X_trn, y_trn, X_val, y_val = train_val_split(X_trn, y_trn, 0)
    print('finish split data')

    hyperparams, drop_features, best_num_round = hyperopt_lightgbm(X_trn, y_trn, X_val, y_val)
    print(f'drop_features: {drop_features}')
    X_trn = X_trn.drop(columns=drop_features)
    X_val = X_val.drop(columns=drop_features)
    to_drop=feature_importance_iter(hyperparams,X_trn,y_trn,X_val,y_val)
    X_trn=X_trn.drop(columns=to_drop)
    print(f'X_trn columns:{X_trn.columns}')
    print(f'X_trn columns:{len(X_trn.columns)}')

    models, num_round_list = train_multiple_models(hyperparams, best_num_round, X_trn, y_trn)
    num_round_list.append(best_num_round)

    final_model = train_all_data(hyperparams, int(np.mean(num_round_list)), X_trn, y_trn)
    #final_model = train_all_data(hyperparams, global_params['lgb_num_round'], X_trn, y_trn)
    print("train all data finished")

    models.append(final_model)
    return lightgbm_predict(models, df_test_data.loc[:, X_trn.columns], df_test_label)


def train_multiple_models(hyperparams, num_rounds, X_train, y_train):
    models = []
    num_round_list = []
    for i in range(0, global_params['model_num'] - 1):
        X_trn, y_trn, X_val, y_val = train_val_split(X_train, y_train, i)
        if global_params['hyperopt_per_mode']:
            hyperparams, drop_features, best_num_round = hyperopt_lightgbm(X_trn, y_trn, X_val, y_val)
        model = train_lightgbm(hyperparams, X_trn, y_trn, X_val, y_val)
        num_round_list.append(model.best_iteration)
        models.append(model)
        print(f"Train model_{i} finished")
    return models, num_round_list

def feature_importance_iter(hyperparams,X_trn,y_trn,X_val,y_val):
    #X_trn,y_trn,X_val,y_val=train_val_split(X_train,y_train,0)
    model=train_lightgbm(hyperparams, X_trn,y_trn,X_val,y_val)
    best_score = model.best_score["valid_0"][global_params['global_metric']]

    importance_df=pd.DataFrame()
    importance_df['feature']=X_trn.columns
    importance_df['importance']=model.feature_importance('gain')
    importance_df=importance_df.sort_values('importance')

    to_drop=[]
    for row in importance_df.iterrows():
        X_trn=X_trn.drop(columns=[row[1]['feature']])
        X_val=X_val.drop(columns=[row[1]['feature']])
        model=train_lightgbm(hyperparams, X_trn,y_trn,X_val,y_val)
        score = model.best_score["valid_0"][global_params['global_metric']]
        if score>=best_score:
            to_drop.append(row[1]['feature'])
            best_score=score
        else:
            break

    print(f'best_score: {best_score}')
    print(f'to_drop_imp_features: {to_drop}')
    return to_drop

def run_on_data(ds_name, percentage):
    train_path = Path(f'./data_split/{ds_name}/{ds_name}Train/{percentage}')
    test_path = Path(f'./data_split/{ds_name}/{ds_name}Test/{percentage}')
    aucs = {}
    for train_file_path in train_path.iterdir():
        test_file_path = test_path / train_file_path.name.replace('train', 'test')
        aucs[train_file_path.name] = test_dataset(ds_name, train_file_path, test_file_path)
    print(aucs)
    utils.write_to_file(aucs, Path(f'./results/{ds_name}_{percentage}_results.json'))


@utils.debug_wrapper
def main():
    ds_name = 'CK'
    per = '30'
    train_path = Path(f'./data_split/{ds_name}/{ds_name}Train/{per}')
    test_path = Path(f'./data_split/{ds_name}/{ds_name}Test/{per}')
    train_file_path = train_path / 'alltrain.csv'
    test_file_path = test_path / 'alltest.csv'
    auc = test_dataset(ds_name, train_file_path, test_file_path)
    print(f'auc: {auc}')


#main()
run_on_data('CK',30)

