def remove_small_variance(df, var_threshold):
    """
    Remove those features with variance less than var_threshold.
    For dataset NASA cm1, searched var_threshold from 0 to 500 and showed 0 gives the best auc.
    But it actually overfits.
    :param var_threshold:
    :return: Dataframe with feature variance > var_threshold
    """
    return df.loc[:, df.var() > var_threshold]
