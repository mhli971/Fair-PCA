from matplotlib import style
from sklearn.decomposition import PCA
from funcs import re, optApprox, loss, mw
from creditProcess import creditProcess
from scipy.linalg import sqrtm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
style.use('seaborn')

M, A, B = creditProcess()
featureNum = 22
# parameters of the mw algorithm
eta = 1  # smaller is finer but needs larger T
T = 10  # larger is better but slower
# overall this 1, 10 combo is good enough

coeff = PCA().fit(M).components_.T  # all components are kept
coeff_A = PCA().fit(A).components_.T
coeff_B = PCA().fit(B).components_.T

loss_A = np.zeros((featureNum, 1))
loss_B = np.zeros((featureNum, 1))

re_A = np.zeros((featureNum, 1))
re_B = np.zeros((featureNum, 1))

re_FairA = np.zeros((featureNum, 1))
re_FairB = np.zeros((featureNum, 1))

re_Ahat = np.zeros((featureNum, 1))
re_Bhat = np.zeros((featureNum, 1))

z_last = np.zeros((featureNum, 1))
z = np.zeros((featureNum, 1))
z_smart = np.zeros((featureNum, 1))

lossFair_max = np.zeros((featureNum, 1))
lossFair_A = np.zeros((featureNum, 1))
lossFair_B = np.zeros((featureNum, 1))

for i in range(featureNum):
    print('Number of features =', i + 1)

    P = np.matmul(coeff[:, :(i + 1)], coeff[:, :(i + 1)].T)

    approx_A = np.matmul(A, P)
    approx_B = np.matmul(B, P)

    # vanilla PCA's average loss on popultion A and B
    loss_A[i] = loss(A, approx_A, i + 1) / A.shape[0]
    loss_B[i] = loss(B, approx_B, i + 1) / B.shape[0]

    re_A[i] = re(A, approx_A) / A.shape[0]
    re_B[i] = re(B, approx_B) / B.shape[0]

    Ahat = optApprox(A, i + 1)
    re_Ahat[i] = re(A, Ahat) / A.shape[0]

    Bhat = optApprox(B, i + 1)
    re_Bhat[i] = re(B, Bhat) / B.shape[0]

    # Fair PCA
    P_fair, z[i], P_last, z_last[i] = mw(A, B, i + 1, eta, T)

    if z[i] < z_last[i]:
        P_smart = P_fair
    else:
        P_smart = P_last

    P_tmp = sqrtm(np.eye(len(P_smart)) - P_smart)
    P_smart = np.eye(len(P_smart)) - P_tmp

    approxFair_A = np.matmul(A, P_smart)
    approxFair_B = np.matmul(B, P_smart)

    lossFair_A = loss(A, approxFair_A, i + 1) / len(A)
    lossFair_B = loss(B, approxFair_B, i + 1) / len(B)
    lossFair_max[i] = max(lossFair_A, lossFair_B)

    re_FairA[i] = re(approxFair_A, A) / A.shape[0]
    re_FairB[i] = re(approxFair_B, B) / B.shape[0]

# plot and save results
if not os.path.exists('figs'):
    os.mkdir('figs')
plt.figure(figsize=(12, 8))
x = [i + 1 for i in range(featureNum)]
plt.xticks(x, fontsize=15)
plt.yticks(fontsize=15)
plt.plot(x, loss_A, 'bv--')
plt.plot(x, loss_B, 'g^--')
plt.plot(x, lossFair_max, 'ro-')
plt.title(
    f'Average loss of PCA and Fair PCA on Credit data', fontsize=20)
plt.legend(['Lower education loss PCA',
           'Higher education loss PCA', 'Fair loss'], fontsize=15)
plt.xlabel('Number of dimensions', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.savefig(f'figs/credit_loss.jpg', dpi=300)

plt.figure(figsize=(12, 8))
x = [i + 1 for i in range(featureNum)]
plt.xticks(x, fontsize=15)
plt.yticks(fontsize=15)
plt.plot(x, re_A, 'r^--')
plt.plot(x, re_B, 'bv--')
plt.plot(x, re_FairA, 'mo-')
plt.plot(x, re_FairB, 'co-')
plt.title(
    'Average reconstruction error (RE) of PCA and Fair PCA on Credit data', fontsize=20)
plt.legend(['Higher education RE PCA', 'Lower education RE PCA', 'Higher education RE Fair PCA',
           'Lower education RE Fair PCA'], fontsize=15)
plt.xlabel('Number of features', fontsize=20)
plt.ylabel('Average reconstruction error (ARE)', fontsize=20)
plt.savefig(f'figs/credit_re.jpg', dpi=300)

if not os.path.exists('tables'):
    os.mkdir('tables')
res_loss = pd.DataFrame(columns=['loss_A', 'loss_B', 'lossFair_max'])
res_loss['loss_A'] = loss_A.squeeze(1)
res_loss['loss_B'] = loss_B.squeeze(1)
res_loss['lossFair_max'] = lossFair_max.squeeze(1)
res_loss.to_csv(f'tables/credit_loss_table.csv')

res_re = pd.DataFrame(
    columns=['re_A', 're_B', 're_FairA', 're_FairB', 're_Ahat', 're_Bhat'])
res_re['re_A'] = re_A.squeeze(1)
res_re['re_B'] = re_B.squeeze(1)
res_re['re_FairA'] = re_FairA.squeeze(1)
res_re['re_FairB'] = re_FairB.squeeze(1)
res_re['re_Ahat'] = re_Ahat.squeeze(1)
res_re['re_Bhat'] = re_Bhat.squeeze(1)
res_re.to_csv(f'tables/credit_re_table.csv')
