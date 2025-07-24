# Import the Python libraries
import time
import random

import numpy
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

def bls_incremental_joint(train_x, train_y, train_xf, train_yf, test_x, test_y, s, C, N1, N2, N3, inputData, m, m2, l):
    TestingAccuracy_z = 0
    f_score = 0
    l = l + 1  # l:?????????

    N11 = N1  # ????????

    # ??????1???????????????????????
    train_err = np.zeros([1, l])
    test_err = np.zeros([1, l])
    train_time = np.zeros([1, l])
    test_time = np.zeros([1, l])

    l2 = np.zeros(l)  # ??????????????

    # ?????????
    time_start = time.time()

    beta11 = []  # ???????????????????

    # ???????????????????????0????1
    train_x = zscore(train_x.transpose(), axis=0, ddof=1).transpose()

    # ????????????????????0.1???????????????????????????(m,n)?m???????n????????????????0.1?
    H1 = np.concatenate((train_x, 0.1 * np.ones((train_x.shape[0], 1))), axis=1)

    y = np.zeros((train_x.shape[0], N2 * N11))  # N2?????????????????????N11??????????????????????

    # ????????????????????
    max_list_set = []
    min_list_set = []

    ### Generation of mapped features
    for i in range(0, N2):
        # ????we (??????, N1),???????????????
        we = 2.0 * np.random.rand(N1, train_x.shape[1] + 1).transpose() - 1.0

        # ??????????????????A1(??????, N1)
        A1 = np.dot(H1, we)
        # ??????????????(0,1)
        [A1, max_list, min_list] = mapminmax(A1)
        del we

        # ?????????????????????? (??????,N1)
        beta1 = sparse_bls(A1, H1, 1e-3, 50).transpose()
        beta11.append(beta1)
        # ??????????????T1 (??????,N1)
        T1 = np.dot(H1, beta1)

        print("Feature nodes in window ", i, ": Max Val of Output ", T1.max(), " Min Val ", T1.min())

        [T1, max_list, min_list] = mapminmax(T1.transpose(), 0, 1)
        T1 = T1.transpose()

        max_list_set.append(max_list)
        min_list_set.append(min_list)

        # ??????????T1?T2?...Tn  (??????,N1*N2)
        y[:, N11 * i: N11 * (i + 1)] = T1  # y:??????????
        # y[:, : N11 * (i + 1)] = T1  # y:??????????

    del H1
    del T1
    del A1

    # Generation of enhancement nodes
    H2 = np.concatenate((y, 0.1 * np.ones((y.shape[0], 1))), axis=1)

    if N1 * N2 >= N3:
        # ????????????????QR????????????
        wh = orth(2 * np.random.rand(N3, N2 * N1 + 1).transpose() - 1)

    else:
        # ??QR?????????
        wh = orth(2 * np.random.rand(N3, N2 * N1 + 1) - 1).transpose()

    Wh = []
    Wh.append(wh)

    T2 = np.dot(H2, wh)  # ????????
    l2[0] = T2.max()
    # ???????
    l2[0] = s * 1.0 / l2[0]

    print("Enhancement nodes: Max Val of Output ", l2, " Min Val ", T2.min())

    # ??????????
    T2 = np.tanh(T2 * l2[0])

    # ???????????y?????????T2?????????T3
    T3 = np.concatenate((y, T2), axis=1)

    del H2
    del T2

    # Moore-Penrose pseudoinverse (function pinv)??????????????
    # ???????????????
    beta = np.dot(pinv(np.dot(T3.transpose(), T3) + np.identity(T3.transpose().shape[0]) * C),
                  T3.transpose())

    # ????????
    beta2 = np.dot(beta, train_y)
    # ??????
    xx = np.dot(T3, beta2)

    # ?????????
    time_end = time.time()
    Training_time = time_end - time_start
    train_time[0][0] = Training_time

    # Training Accuracy
    yy = result(xx)
    train_yy = result(train_y)

    # ??????cnt????????????????????
    cnt = 0
    for i in range(0, len(yy)):
        if yy[i] == train_yy[i]:
            cnt = cnt + 1

    TrainingAccuracy = cnt * 1.0 / train_yy.shape[0]

    train_err[0][0] = TrainingAccuracy

    print("Training Accuracy is : ", TrainingAccuracy * 100, " %")

    ### Testing Process at the beginning of the incremental learning
    # Testing - begin
    time_start = time.time()

    # ????????????
    test_x = zscore(test_x.transpose(), axis=0, ddof=1).transpose()

    HH1 = np.concatenate((test_x, 0.1 * np.ones((test_x.shape[0], 1))), axis=1)
    yy1 = np.zeros((test_x.shape[0], N2 * N11))

    ### Generation of mapped features
    for i in range(0, N2):
        beta1 = beta11[i]

        TT1 = np.dot(HH1, beta1)

        max_list = max_list_set[i]
        min_list = min_list_set[i]

        # ??????????
        [TT1, max_list, min_list] = mapminmax(TT1.transpose(), 0, 1, max_list, min_list)
        TT1 = TT1.transpose()

        del beta1
        del max_list
        del min_list

        yy1[:, N11 * i: N11 * (i + 1)] = TT1
        # yy1[:, : N11 * (i + 1)] = TT1

    del TT1
    del HH1

    ### Generation of enhancement nodes
    HH2 = np.concatenate((yy1, 0.1 * np.ones((yy1.shape[0], 1))), axis=1)

    TT2 = np.tanh(np.dot(HH2, wh) * l2[0])

    TT3 = np.concatenate((yy1, TT2), axis=1)

    # del HH2;
    del wh
    del TT2

    x = np.dot(TT3, beta2)

    time_end = time.time()
    Testing_time = time_end - time_start

    # Testing - end
    test_time[0][0] = Testing_time

    ### Testing accuracy at the beginning of the incremental learning
    y1 = result(x)
    test_yy = result(test_y)

    cnt = 0
    for i in range(0, len(y1)):
        if y1[i] == test_yy[i]:
            cnt = cnt + 1

    TestingAccuracy = cnt * 1.0 / test_yy.shape[0]

    test_err[0][0] = TestingAccuracy

    ### Incremental training steps
    train_y1 = np.zeros((0, 6))
    for e in range(0, l - 1):
        print("Tne number of Incremental Learning times: ", e)

        # ????????
        time_start = time.time()
        # ????????????????????????
        # train_xx = zscore(np.float128(train_xf[(int)(inputData) : (int)(inputData) + (e + 1) * m, : ])
        # .transpose(), axis=0, ddof=1).transpose()
        train_xx = zscore(train_xf[(int)(inputData): (int)(inputData) + (e + 1) * m, :]
                          .transpose(), axis=0, ddof=1).transpose()
        # ???????????????????
        if e == 0:
            train_y11 = train_yf[0:(int)(inputData) + (e + 1) * m, :]
        else:
            train_y11 = train_yf[(int)(inputData): (int)(inputData) + (e + 1) * m, :]
        # train_y11 = train_yf[0:(int)(inputData) + (e + 1) * m, :]
        train_y1 = np.concatenate((train_y1, train_y11), axis=0)

        Hx1 = np.concatenate((train_xx, 0.1 * np.ones((train_xx.shape[0], 1))), axis=1)
        yx = []

        ### Generation of mapped features
        for i in range(0, N2):

            beta1 = beta11[i]

            Tx1 = np.dot(Hx1, beta1)

            max_list = max_list_set[i]
            min_list = min_list_set[i]

            [Tx1, max_list, min_list] = mapminmax(Tx1.transpose(), 0, 1, max_list, min_list)
            Tx1 = Tx1.transpose()

            # ????????????????yx
            if i == 0:
                yx = Tx1
            else:
                yx = np.concatenate((yx, Tx1), axis=1)

        ### Generation of enhancement nodes
        Hx2 = np.concatenate((yx, 0.1 * np.ones((yx.shape[0], 1))), axis=1)
        tx22 = []

        # Concatenate enhancement nodes with added enhancement nodes/step
        for o in range(0, e + 1):
            wh = Wh[o]  # ???????????????wh

            tx2 = np.dot(Hx2, wh)
            tx2 = np.tanh(tx2 * l2[o])

            # ??????????????tx22
            if o == 0:
                tx22 = tx2
            else:
                tx22 = np.concatenate((tx22, tx2), axis=1)

        # Concatenate mapped features with enhancement nodes and added enhancement nodes/step
        # ???????????yx?????????tx22?????????tx2x
        tx2x = np.concatenate((yx, tx22), axis=1)

        # Moore-Penrose pseudoinverse (function pinv)
        # ???????????????????????
        betat = np.dot(pinv(np.dot(tx2x.transpose(), tx2x) + np.identity(tx2x.transpose().shape[0]) * C),
                       tx2x.transpose())

        # ??y:??????????
        beta = np.concatenate((beta, betat), axis=1)
        T3 = np.concatenate((T3, tx2x), axis=0)
        y = np.concatenate((y, yx), axis=0)

        H2 = np.concatenate((y, 0.1 * np.ones((y.shape[0], 1))), axis=1)

        # Generation of random weights for added enhancement nodes/step??????????????
        if N1 * N2 >= m2:
            wh = orth(2 * np.random.rand(m2, N2 * N1 + 1).transpose() - 1)
        else:
            wh = orth(2 * np.random.rand(m2, N2 * N1 + 1) - 1).transpose()

        Wh.append(wh)

        t2 = np.dot(H2, wh)

        l2[e + 1] = t2.max()

        l2[e + 1] = s * 1.0 / l2[e + 1]
        # ????????????t2
        t2 = np.tanh(t2 * l2[e + 1])
        # ???????????T3???????????t2????????????????????T3_Temp
        T3_Temp = np.concatenate((T3, t2), axis=1)
        # ????
        d = np.dot(beta, t2)
        # ????
        c = t2 - np.dot(T3, d)

        # ????????????????c?????
        if np.any(c) != 0:
            # Moore-Penrose pseudoinverse (function pinv)
            b = np.dot(pinv(np.dot(c.transpose(), c) + np.identity(c.transpose().shape[0]) * C),
                       c.transpose())

        else:
            w = d.shape[1]
            # Moore-Penrose pseudoinverse (function pinv)
            # ??? b
            b = np.dot(pinv(np.identity(w) + np.dot(d.transpose(), d)),
                       np.dot(d.transpose(), beta))

        # ????? np.dot(d, b)??????????
        beta = np.concatenate((beta - np.dot(d, b), b), axis=0)

        # ?????????
        beta2 = np.dot(beta, train_y1)

        T3 = T3_Temp

        # ??????
        xx = np.dot(T3, beta2)

        time_end = time.time()
        Training_time = time_end - time_start

        # Incremental training - end
        train_time[0][e + 1] = Training_time

        ### Incremental training Accuracy
        yy = result(xx)
        train_yy = result(train_y1)

        cnt = 0
        for i in range(0, len(train_yy)):
            if yy[i] == train_yy[i]:
                cnt = cnt + 1

        TrainingAccuracy = cnt * 1.0 / train_yy.shape[0]
        train_err[0][e + 1] = TrainingAccuracy

        ### Incremental testing at each step
        # Incremental testing - begin
        time_start = time.time()
        wh = Wh[e + 1]
        tt2 = np.tanh(np.dot(HH2, wh) * l2[e + 1])
        TT3 = np.concatenate((TT3, tt2), axis=1)
        x = np.dot(TT3, beta2)

        time_end = time.time()
        Testing_time = time_end - time_start

        # Incremental testing - end
        test_time[0][e + 1] = Testing_time

        ### Incremental testing accuracy
        y1 = result(x)
        test_yy = result(test_y)

        cnt = 0
        for i in range(0, len(y1)):
            if y1[i] == test_yy[i]:
                cnt = cnt + 1

        TestingAccuracy = cnt * 1.0 / test_yy.shape[0]

        label = test_yy
        predicted = y1

        TestingAccuracy_z = accuracy_score(label, predicted)
        f_score = f1_score(label, predicted, average='macro')
        pre = precision_score(label, predicted, average='macro')
        recall = recall_score(label, predicted, average='macro')

        test_err[0][e + 1] = TestingAccuracy
        print("Test Accuracy: ", TestingAccuracy_z, "\tF-Score: ", f_score, "\tPrecision: ", pre,
              "\tRecall: ", recall)
        print("*********************************************************")
        f1 = open('results/accurancy.txt', 'a+')
        f1.write(str(e) + " " + str(TestingAccuracy_z * 100) + " " + str(f_score) + " " + str(pre) + " " + str(
            recall) + " " +
                 str(sum(train_time[0])) + " " + str(sum(test_time[0])) + " " + str(
            TrainingAccuracy * 100) + " " + '\n')
        print("*********************************************************")

    return TrainingAccuracy, TestingAccuracy_z, sum(train_time[0]), sum(test_time[0]), f_score, pre, recall
#                  TrainingAccuracy, TestingAccuracy, Training_time, Testing_time, f_score
#########################################################################################################
