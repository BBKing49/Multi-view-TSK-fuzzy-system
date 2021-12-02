import pandas as pd
import numpy as np
from numpy.matlib import repmat
from fuzzy_clustering import ESSC,FuzzyCMeans


def preproc(df, k):

    d = len(df.columns)
    n = len(df)
    #
    # U, v = fuzzyCMeansClustering(df, k)
    # U = np.array(U).T
    #
    # b = np.zeros((k, d))
    # for i in range(k):
    #     v1 = repmat(v[i], n, 1)
    #     u = U[i, :]
    #     uu = repmat(u, d, 1)
    #     temp = pow((df-v1), 2)*uu.T
    #     b[i, :] = np.sum( temp, axis=0)/np.sum(uu)/1
    #
    # v = np.array(v)

    fuzzy_cluster = FuzzyCMeans(k).fit(df)
    v = fuzzy_cluster.center_
    b = fuzzy_cluster.variance_
    # U = fuzzy_cluster.train_u

    # b = np.zeros((k, d))
    # for i in range(k):
    #     v1 = repmat(v[i], n, 1)
    #     u = U[i, :]
    #     uu = repmat(u, d, 1)
    #     temp = pow((df-v1), 2)*uu.T
    #     b[i, :] = np.sum( temp, axis=0)/np.sum(uu)/1

    # d = -(np.expand_dims(df, axis=2) - np.expand_dims(centers.T, axis=0)) ** 2 / (2 * delta.T)
    # d = np.exp(np.sum(d, axis=1))
    # d = np.fmax(d, np.finfo(np.float64).eps)

    return v, b

def fromxtoz(df, v, b):

    df = df.reset_index(drop=True)
    n = len(df)
    temp = pd.DataFrame(np.ones((n, 1)))
    df1 = pd.concat((df, temp), axis=1, ignore_index=True)

    k = v.shape[0]
    d0 = v.shape[1]
    wt = np.zeros((n, k))

    for i in range(k):
        v1 = repmat(v[i, :], n, 1)
        bb = repmat(b[i, :], n, 1)
        wt[:, i] = np.exp(-np.sum( pow((df-v1), 2)/bb, axis=1))

    wt2 = np.sum(wt, axis=1)
    wt22 = (repmat(wt2, k, 1))
    wt22 = wt22.T
    wt = wt/wt22

    zt = []
    for i in range(k):
        wt1 = wt[:, i]
        wt2 = repmat(wt1,  d0+1, 1)
        if i==0:
            zt = df1*wt2.T
        else:
            zt = pd.concat((zt, df1*wt2.T), axis=1)

    zt = pd.DataFrame(zt)
    zt.fillna(0.00001)

    # N = X.shape[0]
    # mem = np.expand_dims(mem, axis=1)
    # X = np.expand_dims(np.concatenate((X, np.ones([N, 1])), axis=1), axis=2)
    # X = np.repeat(X, repeats=mem.shape[1], axis=2)
    # xp = X * mem
    # xp = xp.reshape([N, -1])

    return zt
#
# df = np.random.rand(1000,500)
# v, b = preproc(pd.DataFrame(df), 3)
# zt = fromxtoz(pd.DataFrame(df), v, b)
# df = np.random.rand(100,5)

