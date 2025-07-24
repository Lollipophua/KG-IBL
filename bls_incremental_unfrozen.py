import time
import random
import numpy as np
import sys
from scipy.stats import zscore
from scipy.linalg import orth
from numpy.linalg import pinv
from bls.processing.result import result
from bls.processing.sparse_bls import sparse_bls
from sklearn import preprocessing
from bls.processing.mapminmax import mapminmax
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score


def bls_incremental_unfrozen(train_x, train_y, train_xf, train_yf, test_x, test_y, s, C, N1, N2, N3, inputData, m, m2,
                             l):
    TestingAccuracy_z = 0
    f_score = 0
    l = l + 1
    N11 = N1
    train_err = np.zeros([1, l])
    test_err = np.zeros([1, l])
    train_time = np.zeros([1, l])
    test_time = np.zeros([1, l])

    l2 = np.zeros(l)
    time_start = time.time()

    beta11 = []
    train_x = zscore(train_x.transpose(), axis=0, ddof=1).transpose()
    H1 = np.concatenate((train_x, 0.1 * np.ones((train_x.shape[0], 1))), axis=1)
    y = np.zeros((train_x.shape[0], N2 * N11))

    max_list_set = []
    min_list_set = []

    ### Generation of mapped features
    for i in range(0, N2):
        we = 2.0 * np.random.rand(N1, train_x.shape[1] + 1).transpose() - 1.0
        A1 = np.dot(H1, we)
        [A1, max_list, min_list] = mapminmax(A1)
        del we
        beta1 = sparse_bls(A1, H1, 1e-3, 50).transpose()
        beta11.append(beta1)
        T1 = np.dot(H1, beta1)

        print("Feature nodes in window ", i, ": Max Val of Output ", T1.max(), " Min Val ", T1.min())

        [T1, max_list, min_list] = mapminmax(T1.transpose(), 0, 1)
        T1 = T1.transpose()
        max_list_set.append(max_list)
        min_list_set.append(min_list)
        y[:, N11 * i: N11 * (i + 1)] = T1

    del H1
    del T1
    del A1

    # Generation of enhancement nodes
    H2 = np.concatenate((y, 0.1 * np.ones((y.shape[0], 1))), axis=1)
    Wh = []
    l2 = np.zeros(l)
    beta = np.dot(pinv(np.dot(H2.transpose(), H2) + np.identity(H2.transpose().shape[0]) * C), H2.transpose())
    beta2 = np.dot(beta, train_y)
    Wh.append(beta)
    l2[0] = 1

    for e in range(0, l - 1):
        print("Incremental Learning times: ", e)
        time_start = time.time()

        # Get new training data
        train_xx = zscore(train_xf[((int)(inputData) + e * m): (int)(inputData) + (e + 1) * m, :].transpose(), axis=0,
                          ddof=1).transpose()
        train_yx = train_yf[(int)(inputData) + e * m + 1: (int)(inputData) + (e + 1) * m, :]
        train_y1 = train_yf[0:(int)(inputData) + (e + 1) * m, :]

        # Generate mapped features
        Hx1 = np.concatenate((train_xx, 0.1 * np.ones((train_xx.shape[0], 1))), axis=1)
        yx = []
        for i in range(0, N2):
            beta1 = beta11[i]
            Tx1 = np.dot(Hx1, beta1)
            [Tx1, max_list, min_list] = mapminmax(Tx1.transpose(), 0, 1, max_list_set[i], min_list_set[i])
            Tx1 = Tx1.transpose()
            if i == 0:
                yx = Tx1
            else:
                yx = np.concatenate((yx, Tx1), axis=1)

        # Generate enhancement nodes
        Hx2 = np.concatenate((yx, 0.1 * np.ones((yx.shape[0], 1))), axis=1)
        tx22 = []
        for o in range(0, e + 1):
            wh = Wh[o]
            tx2 = np.dot(Hx2, wh)
            tx2 = np.tanh(tx2 * l2[o])
            if o == 0:
                tx22 = tx2
            else:
                tx22 = np.concatenate((tx22, tx2), axis=1)

        # Train and test
        beta = np.dot(pinv(np.dot(tx22.transpose(), tx22) + np.identity(tx22.transpose().shape[0]) * C),
                      tx22.transpose())
        beta2 = np.dot(beta, train_yx)
        predicted = np.dot(tx22, beta2)

        # Calculate testing metrics
        TestingAccuracy = accuracy_score(train_yx, predicted)
        f_score = f1_score(train_yx, predicted, average='macro')
        pre = precision_score(train_yx, predicted, average='macro')
        recall = recall_score(train_yx, predicted, average='macro')

        test_err[0][e + 1] = TestingAccuracy

        # Update other metrics
        train_err[0][e + 1] = 1 - TestingAccuracy
        train_time[0][e + 1] = time.time() - time_start
        test_time[0][e + 1] = 0
        l2[e + 1] = 1

    return TestingAccuracy_z, f_score, pre, recall, train_err, test_err, train_time, test_time
