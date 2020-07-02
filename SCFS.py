import numpy as np
import math
import skfuzzy as fuzz
import time


def scfs(X, **kwargs):
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    gamma = 1e+6

    XX = kwargs['XX']
    XTX = kwargs['XTX']
    c = kwargs['n_clusters']
    n, p = X.shape

    G = cmeans_initialization(X, n, c)
    GG = np.dot(G, np.transpose(G))
    GTG = np.dot(np.transpose(G), G)

    D = np.identity(p)
    one = np.ones((n, n), dtype=np.float64)

    maxIter = 1000
    obj = np.zeros(maxIter)

    for iter_step in range(maxIter):
        T = np.linalg.inv(alpha * XTX + beta * D + 1e-6 * np.eye(p))
        W = np.dot(T, alpha * np.dot(np.transpose(X), G))
        temp = np.sqrt((W*W).sum(1))
        temp[temp < 1e-16] = 1e-16
        temp = 0.5 / temp
        D = np.diag(temp)

        M = np.dot(XX + n*gamma*one, G)

        num = 2 * M + alpha * np.dot(X, W)
        den = np.dot(M, GTG) + np.dot(GG, M) + alpha * G

        T0 = np.divide(num, den)
        G = G * np.array(T0)

        GG = np.dot(G, np.transpose(G))
        GTG = np.dot(np.transpose(G), G)
        GGone = np.dot(GG, one)

        obj[iter_step] = np.linalg.norm(X - np.dot(GG, X), 'fro')**2 + alpha * np.linalg.norm(np.dot(X, W) - G, 'fro')**2 + beta * (np.sqrt((W*W).sum(1))).sum() + gamma * np.linalg.norm(GGone-one, 'fro')**2
        if iter_step >= 1 and math.fabs(obj[iter_step] - obj[iter_step-1])/obj[iter_step] < 1e-5:
            break

    return W


def cmeans_initialization(X, n_samples, n_clusters):
    cntr, Y, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.transpose(), n_clusters, 1.0005, error=0.0001, maxiter=300, init=None)
    Y = np.transpose(Y)

    G = np.zeros((n_samples, n_clusters), dtype=np.float64)

    for i in range(n_samples):
        for j in range(n_clusters):
            G[i, j] = (Y[i, j] / np.sqrt(np.sum(Y[:, j])))

    return G

