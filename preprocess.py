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
