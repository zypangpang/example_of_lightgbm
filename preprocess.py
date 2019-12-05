import sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat


# args=sys.argv[1:]
# print(args)

def trans_data(rPath, wPath):
    # x = loadmat('data_raw/NASA/NASATrain/kc3train.mat')
    x = loadmat(str(rPath))
    data = x[rPath.stem]
    col_name = list(map(lambda x: f'n_{x}', list(range(0, len(data[0][0][0])))))
    col_name[-1] = 'l'
    data = np.concatenate(data[0])
    df = pd.DataFrame(data=data, columns=col_name)
    with wPath.open('w') as f:
        f.write(df.to_csv(index=False))


def trans_mat_to_csv():
    rp = Path('./data_raw/')
    wp = Path("./data/")
    for dir1 in rp.iterdir():
        for dir2 in dir1.iterdir():
            for file in dir2.iterdir():
                wpath = wp / file.relative_to('data_raw/').with_suffix('.csv')
                trans_data(file, wpath)


def process_na(df):
    if df.isnull().sum().any():
        df.fillna('0')


def merge_datasets():
    def merge_helper(dir_path):
        df_all = None
        for file in dir_path.iterdir():
            df = pd.read_csv(str(file))
            if df_all is None:
                df_all = df
            else:
                df_all = df_all.append(df, ignore_index=True)
        return df_all

    def write_csv(path, df):
        with path.open('w') as f:
            f.write(df.to_csv(index=False))

    dataset_names = ['NASA', 'CK']
    for name in dataset_names:
        ds_train_path = Path(f"./data/{name}/{name}Train/")
        ds_test_path = Path(f"./data/{name}/{name}Test/")
        df_train = merge_helper(ds_train_path)
        df_test = merge_helper(ds_test_path)
        out_train_path = ds_train_path / 'alltrain.csv'
        out_test_path = ds_test_path / 'alltest.csv'
        write_csv(out_train_path, df_train)
        write_csv(out_test_path, df_test)


def process_extra_label(df,remove):
    if df['l'].nunique() > 2:
        if remove:
            to_drop_rows = df.index[(df['l'] != -1) & (df['l'] != 1)]
            return df.drop(to_drop_rows)
        else:
            df['l'].loc[(df['l'] != -1) & (df['l'] != 1)] = 1
            return df
    return df
