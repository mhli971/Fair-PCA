import pandas as pd
from sklearn.decomposition import PCA
from numpy.linalg import norm
import numpy as np


def re(Y, Z):
    '''
    Calculate the reconstruction error of matrix Y with respect to matrix Z
    Matrix Y and Z are of the same size
    '''

    return norm(Y - Z, ord='fro') ** 2


def optApprox(M, d):
    '''
    optimal approximation by pca
    d is the number of components
    '''
    pca = PCA(n_components=d)
    pca.fit(M)
    P = np.matmul(pca.components_.T, pca.components_)
    Mhat = np.matmul(M, P)

    return Mhat


def loss(Y, Z, d):
    '''
    loss of matrix Y with respect to matrix Z
    Y and Z are of the same size
    '''
    Yhat = optApprox(Y, d)
    loss = re(Y, Z) - re(Y, Yhat)

    return loss


def oracle(n, A, m_A, B, m_B, alpha, beta, d, w_1, w_2):
    '''
    Given an input matrix_A= A^T A, matrix_B=B^T B both of size n by n, d, and weights w_1,w_2,
    solve the optimization problem:
    min w_1 z_1 + w_2 z_2 s.t.
    z_1 >= alpha - <matrix_A , P>
    z_2 >= beta - <matrix_B , P>
    tr(P) <= d
    0 <= P <= I
    '''
    if A.shape != (m_A, n) or B.shape != (m_B, n):  # wrong size
        print('Input matrix to oracle method has wrong size. Set P, l_1, l_2 to be 0')
        P_o = 0
        z_1 = 0
        z_2 = 0

    covA = np.matmul(A.T, A)
    covB = np.matmul(B.T, B)

    # We weight A ^ T A by w_1 and B ^ T B by w_2. Note that A ^ T A = summation of
    # v_i v_i ^ T over vector v_i in group A, so w_1 A ^ T A can be obtained by
    # scaling each v_i to sqrt(w_1) v_i. Similar for group B.

    pca = PCA(n_components=d)
    A_wgted = (np.sqrt((1 / m_A) * w_1)) * A
    B_wgted = (np.sqrt((1 / m_B) * w_2)) * B
    M_tmp = np.concatenate([A_wgted, B_wgted], axis=0)
    pca.fit(M_tmp)
    coeff_P_o = pca.components_

    # coeff_P_o is now an n x d matrix
    P_o = np.matmul(coeff_P_o.T, coeff_P_o)
    z_1 = (1 / m_A) * (alpha - sum(sum(covA * P_o)))  # all sum
    z_2 = (1 / m_B) * (beta - sum(sum(covB * P_o)))

    return P_o, z_1, z_2


def mw(A, B, d, eta, T):
    '''
    matrix B has the points in group B as its rows
    matrix A has the points in group A as its rows
    population A and B are expected to be normalized to have mean 0. 
    d is the target dimension
    eta and T are MW's parameters
    '''
    print('MW method is called')

    covA = np.matmul(A.T, A)
    covB = np.matmul(B.T, B)

    # m_A and m_B are size of data set A and B respectively
    m_A = A.shape[0]
    m_B = B.shape[0]
    n = A.shape[1]  # num features

    Ahat = optApprox(A, d)
    alpha = norm(Ahat, ord='fro') ** 2

    Bhat = optApprox(B, d)
    beta = norm(Bhat, ord='fro') ** 2

    # MW

    # start with uniform weight
    w_1 = 0.5
    w_2 = 0.5

    # P is our answer, so I keep the sum of all P_t along the way
    P = np.zeros((n, n))
    # just for record at the end to see the progress over iterates
    record = pd.DataFrame(
        columns=["iteration", "w_1", "w_2", "loss A", "loss B",
                 "loss A by average", "loss B by average"],
        index=range(T))

    for t in range(T):

        # think of P_tmp as P_t we got by weighting with w_1,w_2
        P_tmp, z_1, z_2 = oracle(
            n, A, m_A, B, m_B, alpha, beta, d, w_1, w_2)
        # z_1, z_2 are losses for group A and B respectively.
        # If z_i is big, group i is bottle neck,
        # so weight group i more next time
        w_1star = w_1 * np.exp(eta * z_1)
        w_2star = w_2 * np.exp(eta * z_2)

        # renormalize
        w_1 = w_1star / (w_1star + w_2star)
        w_2 = w_2star / (w_1star + w_2star)

        # add to the sum of P_t
        P = P + P_tmp

        # record the progress
        P_average = (1 / (t + 1)) * P
        loss_A_avg = (1 / m_A) * (alpha - sum(sum(covA * P_average)))
        loss_B_avg = (1 / m_B) * (beta - sum(sum(covB * P_average)))
        record.iloc[t, :] = [t, w_1, w_2, z_1, z_2, loss_A_avg, loss_B_avg]

    # take average of P_t
    P = (1 / T) * P

    # calculate loss of P_average
    z_1 = 1 / m_A * (alpha - sum(sum(covA * P)))
    z_2 = 1 / m_B * (beta - sum(sum(covB * P)))
    z = max(z_1, z_2)

    # in case last iterate is preferred to the average
    P_last = P_tmp

    # calculate loss of P_average
    zl_1 = 1 / m_A * (alpha - sum(sum(covA * P_last)))
    zl_2 = 1 / m_B * (beta - sum(sum(covB * P_last)))
    z_last = max(zl_1, zl_2)

    print('MW method is finished. The loss for group A is ',
          z_1, '; For group B is', z_2)
    print(record)
    return P, z, P_last, z_last
