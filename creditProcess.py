import pandas as pd


def creditProcess():

    # preprocess the credit data.
    # The output of the function is the centered data as matrix M. Must be centered!
    # Centered low educated group A and high educated as group B.

    data = pd.read_csv('data/credit/default_degree.csv',
                       index_col=None, header=1).drop(['ID'], axis=1)

    # vector of sensitive attribute: education
    sensitive = data.loc[:, 'EDUCATION'].copy()

    # normalizing the sensitive attribute vector
    # to have 0 for grad school and university level education and positive value for high school, other
    normalized = (sensitive - 1) * (sensitive - 2)

    # drop the colum corresponding to the senstive attribute.
    # data = data.drop(['EDUCATION', 'PAY_AMT6'], axis=1)
    # note: I dropped PAY_AMT6 to replicate the paper results, but this might be a mistake
    data = data.drop(['EDUCATION'], axis=1)
    # I decided to not drop PAY_AMT6 after inspection, the authors might have made a mistake

    # centering the data and normalizing the variance across each column
    # this is the difference, I debugged for 1 hour for it
    data = (data - data.mean()) / data.std(ddof=1)

    # data for low educated populattion
    lowEd_copy = data.loc[normalized != 0, :].copy()

    # date for high educated population
    highEd_copy = data.loc[normalized == 0, :].copy()

    # centering data for high - and low-educated
    lowEd_copy = lowEd_copy - lowEd_copy.mean()
    highEd_copy = highEd_copy - highEd_copy.mean()

    M = data.values
    A = lowEd_copy.values
    B = highEd_copy.values

    return M, A, B
