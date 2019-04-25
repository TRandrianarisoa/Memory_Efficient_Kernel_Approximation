import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import copy
from sklearn.cluster import KMeans
from scipy.linalg import block_diag
from scipy.sparse.linalg import svds


###################### Kernel COmputation ##########################################

def RBF_kernel(x, y, gamma):
    """
    :param x: vector of size d
    :param y: vector of size d
    :param gamma: scale parameter
    :return: RBF Kernel between x and y with scaling parameter gamma
    """
    return np.exp(-gamma*((x-y)**2).sum())


def get_gram_matrix(dataset, kernel, **param):
    """
    :param dataset: n x d array corresponding to n data points of size d
    :param kernel: kernel function to use
    :param param: parameters dict for the kernel function
    :return: Gram matrix of the dataset
    """
    n = dataset.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = kernel(dataset[i], dataset[j], **param)
    return G


###################### Tests on a subsample ##########################################

def extract_random_rows(dataset, nrows):
    """
    :param dataset: n x d array for n data points of size d
    :param nrows: number of rows to select
    :return: A subset with nrows data points
    """
    r = np.arange(dataset.shape[0])
    indices = np.random.choice(r, size=nrows, replace=False)
    return dataset[indices]


def plot_G_sub(data, nrows, kernel, **param):
    """
    :param data: Dataset of size n x d
    :param nrows: Number of rows to extract for the figure
    :param kernel: Kernel function to use for the Gram matrix
    :param param: Parameters of the kernel function
    :return: Figure of a colormap of a Gram matrix formed by extracting nrows data points
    of the original dataset. The figure is in gray scale
    """
    X = extract_random_rows(data, nrows)
    G = get_gram_matrix(X, kernel, **param)
    plt.imshow(G, cmap='gray')
    plt.show()


def plot_clust_G_sub(data, nrows, kernel, clust_alg, **param):
    """
    :param data: Dataset of size n x d
    :param nrows: Number of rows to extract for the figure
    :param kernel: Kernel function to use for the Gram matrix
    :param clust_alg: A pre-trained clustering model. Must have a predict function (sklearn-like)
    :param param: Parameters of the kernel function
    :return: Figure of a colormap of a Gram matrix formed by extracting nrows data points
    of the original dataset. Data are regrouped by clusters, according to the clustering model
    set as input. The figure is in gray scale
    """
    X = extract_random_rows(data, nrows)
    pred = clust_alg.predict(X)
    group_indices = np.argsort(pred)
    G = get_gram_matrix(X[group_indices], kernel, **param)
    plt.imshow(G, cmap='gray')
    plt.show()


####################### Evaluation of the outputs of the algorithm ########################

def compute_distance(G1, G2):
    """
    :param G1: 2-D array
    :param G2: 2-D array of same size as G1
    :return: Square of the Frobenius norm of G2-G1
    """
    return ((G1-G2)**2).sum()


def build_G_from_MEKA(W, diag_L, L):
    """
    :param W: List of the diagonal blocks of W computed in MEKA approximation
    :param diag_L: List of the C diagonal blocks of L in MEKA, formed using Nyström approximation
    :param L: List of C lists containing the C-1 blocks formed by MEKA, without the diagonal
    block (please refer to the output of our MEKA function for better understanding)
    :return: The Gram matrix obtained by processing the outputs of MEKA algorithm
    """
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