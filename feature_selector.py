import numpy as np
def remove_small_variance(df, var_threshold):
    """
    Remove those features with variance less than var_threshold.
    For dataset NASA cm1, searched var_threshold from 0 to 500 and showed 0 gives the best auc.
    But it actually overfits.
    :param var_threshold:
    :return: Dataframe with feature variance > var_threshold
    """
    return df.loc[:, df.var() > var_threshold]

def remove_many_na_col(df, na_threshold):
    na_fractions=1-df.count()/len(df.index)
    return df.drop(columns=df.columns[na_fractions>=na_threshold])

def remove_collinear_col(df,col_threshold):
    corr_matrix=df.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
    return df.drop(columns=upper.columns[(upper.abs()>col_threshold).any()])

def remove_single_unique(df):
    to_drop=df.columns[df.nunique()==1]
    print(f"single unique: {to_drop}")
    return df.drop(columns=to_drop)
