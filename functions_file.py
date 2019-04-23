import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import bz2
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import copy
from sklearn.cluster import KMeans
from scipy.linalg import block_diag
from scipy.sparse.linalg import svds


def RBF_kernel(x, y, gamma):
    return np.exp(-gamma*((x-y)**2).sum())


def get_gram_matrix(dataset, kernel, **param):
    n = dataset.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = kernel(dataset[i], dataset[j], **param)
    return G


def extract_random_rows(dataset, nrows):
    r = np.arange(dataset.shape[0])
    indices = np.random.choice(r, size=nrows, replace=False)
    return(dataset[indices])


def plot_G_sub(data, nrows, kernel, **param):
    X = extract_random_rows(data, nrows)
    G = get_gram_matrix(X, kernel, **param)
    plt.imshow(G, cmap='gray')
    plt.show()


def plot_clust_G_sub(data, nrows, kernel, clust_alg, **param):
    X = extract_random_rows(data, nrows)
    pred = clust_alg.predict(X)
    group_indices = np.argsort(pred)
    G = get_gram_matrix(X[group_indices], kernel, **param)
    plt.imshow(G, cmap='gray')
    plt.show()


# Evaluation of a Gram Matrix Approximation
def compute_distance(G1, G2):
    return ((G1-G2)**2).sum()


# Block Kernel Approximation
def BKA(X, C_pred, kernel, **param):
    n_clusters = C_pred.max()+1
    blocks = []
    for c in tqdm(range(n_clusters)):
        ind = np.where(C_pred == c)
        blocks.append(get_gram_matrix(X[ind], kernel, **param))
    return block_diag(*blocks)


def objective(G, C):
    obj = 0
    n_cluster = C.max()+1
    for c in range(n_cluster):
        ind = np.where(C == c)
        obj += (G[ind]**2).sum()/ind[0].shape[0]
    return obj


def run_BKA(data, ncluster, njobs, kernel, **param):
    KM = KMeans(n_clusters=ncluster, n_jobs=njobs)
    KM.fit(data)
    C = KM.predict(data)
    ind = np.argsort(C)
    X, C = data[ind], C[ind]
    BK = BKA(X, C, kernel, **param)
    return BK, C


def find_best_BKA(data, kernel, c_min, c_max, step, njobs, **param):
    obj, obj_list = 0, []
    current_max = c_min
    for c in np.arange(c_min, c_max+1, step):
        BK, C = run_BKA(data, ncluster=c, njobs=njobs, kernel=kernel, **param)
        new_obj = objective(BK, C)
        obj_list.append(new_obj)
        if new_obj > obj:
            current_max = c
            obj = new_obj
    return current_max, obj_list


# Nyström Method
def get_sub_G(sub1, sub2, kernel, **param):
    """
    Return a matrix with kernel evaluation between each element
    of sub1 and each element of sub2
    """
    n, npp = sub1.shape[0], sub2.shape[0]
    sub_G = np.zeros((n, npp))
    for i in range(n):
        for j in range(npp):
            sub_G[i, j] = kernel(sub1[i], sub2[j], **param)
    return sub_G


def Nystrom(data, m, k, kernel, return_decompo=True, **param):
    n = data.shape[0]
    sample_ind = np.random.choice(np.arange(n), np.min([n, m]), replace=False)
    sample = data[sample_ind]
    C = get_sub_G(data, sample, kernel, **param)
    G = get_gram_matrix(sample, kernel, **param)
    U, s, Ut = svds(G, k)
    M = np.linalg.pinv(U @ np.diag(s) @ Ut)
    if return_decompo:
        return C, M
    return C @ M @ C.T


# MEKA
def solve_lsp(Ws, Wt, Gst):
    l_term = np.linalg.inv(Ws.T @ Ws + 0.001*np.eye(Ws.shape[1]))
    r_term = np.linalg.inv(Wt.T @ Wt + 0.001*np.eye(Wt.shape[1]))
    return l_term @ Ws.T @ Gst @ Wt @ r_term


def MEKA(data, C, m, k, rho, kernel, **param):
    n, n_cluster = data.shape[0], int(C.max())+1
    cl_size, cl_ind = [], []
    diag_L, diag_W, all_L = [], [], []
    for i in range(n_cluster):
        ind = np.where(C == i)[0]
        cl_size.append(len(ind))
        cl_ind.append(ind)
        W, L = Nystrom(data[ind], m, k, kernel, **param)
        diag_L.append(L)
        diag_W.append(W)
    for i in range(n_cluster):
        l = []
        for j in range(n_cluster):
            if i != j:
                size_i, size_j = np.min([cl_size[i], k*(1+rho)]), np.min([cl_size[j], k*(1+rho)])
                vs = np.random.choice(np.arange(cl_size[i]), size_i, replace=False)
                vt = np.random.choice(np.arange(cl_size[j]), size_j, replace=False)
                vs_data = cl_ind[i][vs]
                vt_data = cl_ind[j][vt]
                g = get_sub_G(data[vs_data], data[vt_data], kernel, **param)
                new_wl, new_wr = diag_W[i][vs], diag_W[j][vt]
                l.append(solve_lsp(new_wl, new_wr, g))
        all_L.append(l)
    return diag_W, diag_L, all_L


def build_G_from_MEKA(W, diag_L, L):
    L_blocks = []
    for i in range(len(W)):
        l = []
        count = 0
        for j in range(len(W)):
            if i == j:
                l.append(diag_L[i])
            else:
                l.append(L[i][count])
                count += 1
        L_blocks.append(l)
    LB = np.block(L_blocks)
    W = block_diag(*W)
    return W @ LB @ W.T


if __name__ == '__main__':
    X_train, y_train = load_svmlight_file('dataset/ijcnn1.bz2')
    X_train = X_train.A
    X_rescale = preprocessing.scale(X_train)
    KM = KMeans(n_clusters=10, n_jobs=-1)
    KM.fit(X_rescale)

    X = extract_random_rows(X_rescale, nrows=1000)
    C = KM.predict(X)
    ind = np.argsort(C)
    X, C = X[ind], C[ind]
    G_X = get_gram_matrix(X, RBF_kernel, **{'gamma': 5e-2})
    res = MEKA(X, C, 50, 15, 2, RBF_kernel, **{'gamma': 5e-2})
    MEKA_X = build_G_from_MEKA(*res)

    print(MEKA_X)
    plt.imshow(G_X, cmap='gray')
    plt.show()
    plt.imshow(MEKA_X, cmap='gray')
    plt.show()