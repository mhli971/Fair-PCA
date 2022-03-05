from datetime import datetime
import pandas as pd
import numpy as np


def ycProcess():

    data = pd.read_csv('data/YCdata/daily-treasury-rates.csv')
    data['Date'] = data['Date'].astype('datetime64')

    # Restrict to data since 1995
    data = data[data['Date'] >= datetime.strptime("1995-01-01", "%Y-%m-%d")]
    # We also remove a few broken data points(all rates are NAN)
    data = data[data.drop("Date", axis=1).T.astype(float).sum(axis=0) != 0]

    # We will remove the 2 month rates, as 8-week treasuries have only been around since October 2018
    # We remove 1 month and 30 year rates as well, since there are more than 10 % NAN values.
    data = data.drop(['Date', '1 Mo', '2 Mo', '30 Yr'], axis=1)

    # only 3 samples of 3 Month is NAN, just ffill them
    data['3 Mo'] = data['3 Mo'].fillna(method='ffill')
    assert data.isna().sum().sum() == 0.0

    def _is_normal(row):  # filter normal yield curves
        if np.mean(np.sort(row) == row) == 1.:
            return True
        else:
            return False

    idx_normal = data.apply(_is_normal, axis=1)

    # centering the data and normalizing the variance across each column
    data = (data - data.mean()) / data.std(ddof=1)

    non_normal = data.loc[idx_normal == False, :].copy()
    normal = data.loc[idx_normal == True, :].copy()

    # centering data
    non_normal = non_normal - non_normal.mean()
    normal = normal - normal.mean()

    M = data.values
    A = non_normal.values
    B = normal.values

    return M, A, B
