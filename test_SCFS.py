import numpy as np
import scipy.io
from SCFS import scfs


def main():

    # Please cite "Unsupervised Feature Selection based on Adaptive Similarity Learning and Subspace Clustering", Mohsen Ghassemi Parsa, Hadi Zare, Mehdi Ghatee

    data_name = 'Lung'
    print data_name

    mat = scipy.io.loadmat(data_name)
    X = mat['X']
    X = X.astype(float)
    X += np.fabs(np.min(X))
    y = mat['Y']
    y = y[:, 0]
    Parm = [1e-4, 1e-2, 1, 1e+2, 1e+4]

    n, p = X.shape
    c = len(np.unique(y))

    XX = np.dot(X, np.transpose(X))
    XTX = np.dot(np.transpose(X), X)

    count = 0
    idx = np.zeros((p, 25), dtype=np.int)

    for Parm1 in Parm:
        for Parm2 in Parm:
            W = scfs(X, XX=XX, XTX=XTX, n_clusters=c, alpha=Parm1, beta=Parm2)
            T = (W * W).sum(1)
            index = np.argsort(T, 0)
            idx[0:p, count] = index[::-1]
            count += 1

if __name__ == '__main__':
    main()
