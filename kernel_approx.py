from utils import *


######################## BKA ####################################################

def BKA(X, C_pred, kernel, **param):
    """
    :param X: Dataset of size n x d
    :param C_pred: Predicted cluster for each point in X (in the same order)
    :param kernel: Kernel function to use
    :param param: Parameters of the kernel function
    :return: Block Kernel Approximation of a matrix, with given predicted clusters
    """
    n_clusters = C_pred.max()+1
    blocks = []
    for c in tqdm(range(n_clusters)):
        ind = np.where(C_pred == c)
        blocks.append(get_gram_matrix(X[ind], kernel, **param))
    return block_diag(*blocks)


def run_BKA(data, ncluster, njobs, kernel, **param):
    """
    :param data: Dataset of size n x d
    :param ncluster: Number of clusters for the K-Means algorithm
    :param njobs: Number of cores to use in K-Means training
    :param kernel: Kernel to use
    :param param: Parameters of the kernel
    :return: Predicted clusters for each data point and corresponding
    Block Kernel Approximation of the input matrix
    """
    KM = KMeans(n_clusters=ncluster, n_jobs=njobs)
    KM.fit(data)
    C = KM.predict(data)
    ind = np.argsort(C)
    X, C = data[ind], C[ind]
    BK = BKA(X, C, kernel, **param)
    return BK, C


def objective(G, C):
    """
    :param G: Some approximation of a Gram matrix
    :param C: Predicted cluster of each data point used to compute G
    :return: Value of the objective of a given approximation according to the repartition of the
    data points in the cluster. This function is used to compare approximations made
     with different numbers of clusters
    """
    obj = 0
    n_cluster = C.max()+1
    for c in range(n_cluster):
        ind = np.where(C == c)
        obj += (G[ind]**2).sum()/ind[0].shape[0]
    return obj


def find_best_BKA(data, kernel, c_min, c_max, step, njobs, **param):
    """
    :param data: Dataset of size n x d
    :param kernel: Kernel function to use to compute the Gram matrix
    :param c_min: Minimal Number of clusters
    :param c_max: Maximal Number of clusters
    :param step: Step to increase the number of clusters between c_min and c_max
    :param njobs: Number of cores to use for each K-Means training
    :param param: Parameters of the Kernel Function
    :return: Optimal number of clusters according to the objective (as defined in the paper),
    list of the values of the objective for each tested value
    """
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


######################## Nyström ###################################################

def get_sub_G(sub1, sub2, kernel, **param):
    """
    :param sub1: a Sub-sample of some dataset
    :param sub2: Another sub-sample of the same dataset
    :param kernel: Kernel Function
    :param param: Parameters of the kernel function
    :return: Matrix containing the Kernel evaluations between each point of sub1
    and each point of sub2
    """
    n, npp = sub1.shape[0], sub2.shape[0]
    sub_G = np.zeros((n, npp))
    for i in range(n):
        for j in range(npp):
            sub_G[i, j] = kernel(sub1[i], sub2[j], **param)
    return sub_G


def Nystrom(data, m, k, kernel, sample='random', return_decompo=True, **param):
    """
    :param data: Dataset of size n x d
    :param m: Number of samples used to compute the approximation
    :param k: Target rank of the low-rank factorization
    :param kernel: Kernel Function
    :param return_decompo: If the algorithm should return the two matrix involved in the decomposition
    (more memory efficient) or the entire approximated kernel matrix
    :param param: Parameters of the kernel
    :return: Nyström approximation of the Gram matrix of the dataset, using m samples and a rank-k factorization
    """
    if sample=='random':
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

######################## Ensemble Nystrom ###################################################

def EnsembleNystrom(data, m, k, kernel, n_experts, **param):
    """
    :param data: Dataset of size n x d
    :param m: Number of samples used to compute the approximation
    :param k: Target rank of the low-rank factorization
    :param kernel: Kernel Function
    :param n_experts: Number of experts aggregated
    :param param: Parameters of the kernel
    :return: Ensemble Nyström approximation of the Gram matrix of the dataset, with all experts having the
     same weight as suggested in the original paper (Kumar, Mohri and Talwalkar, 2009).
    """
    n = data.shape[0]
    sample_ind = np.random.choice(np.arange(n), np.min([n, n_experts*m]), replace=False)
    sample_ind = np.array_split(sample_ind, n_experts)
    sample_ind = [data[samp] for samp in sample_ind]
    return np.mean([Nystrom(data, m, k, kernel, sample_ind[i], return_decompo=False, **param)
                    for i in range(n_experts)])


######################## KMeans Nystrom ###################################################

def KMEANystrom(data, m, k, kernel, return_decompo=True, **param):
    """
    :param data: Dataset of size n x d
    :param m: Number of samples used to compute the approximation
    :param k: Target rank of the low-rank factorization
    :param kernel: Kernel Function
    :param n_experts: Number of experts aggregated
    :param param: Parameters of the kernel
    :return: Ensemble Nyström approximation of the Gram matrix of the dataset, with all experts having the
     same weight as suggested in the original paper (Kumar, Mohri and Talwalkar, 2009).
    """
    n = data.shape[0]
    if n > 20000: # following the suggestion from Zhang et al., 2012
        sample_ind = np.random.choice(np.arange(n), 20000, replace=False)
        sample = data[sample_ind]
        centroids = KMeans(n_clusters=m).fit(sample).cluster_centers_
    else:
        centroids = KMeans(n_clusters=m).fit(data).cluster_centers_
    C = get_sub_G(data, centroids, kernel, **param)
    G = get_gram_matrix(centroids, kernel, **param)
    U, s, Ut = svds(G, k)
    M = np.linalg.pinv(U @ np.diag(s) @ Ut)
    if return_decompo:
        return C, M
    return C @ M @ C.T


######################## MEKA ###################################################

def solve_lsp(Ws, Wt, Gst):
    """
    :param Ws: Array of size n x k
    :param Wt: Array of size n x k
    :param Gst: Array of size k x k
    :return: Solution of the Least Squares Problem used in the MEKA algorithm
    """
    l_term = np.linalg.inv(Ws.T @ Ws + 0.001*np.eye(Ws.shape[1]))
    r_term = np.linalg.inv(Wt.T @ Wt + 0.001*np.eye(Wt.shape[1]))
    return l_term @ Ws.T @ Gst @ Wt @ r_term


def MEKA(data, C, m, k, rho, kernel, **param):
    """
    :param data: Dataset of size n x k
    :param C: Number of clusters/diagonal blocks in W
    :param m: Number of samples for each cluster
    :param k: Rank of the Nyström approximation of each kernel block
    :param rho: Multiplier for the number of samples used in each off-diagonal block
    :param kernel: Kernel Function
    :param param: Parameters of the kernel function
    :return: diag_W: list of the C diagonal blocks in W, diag_L: list of the C diagonal
    blocks in L. all_L: 2-D List of off-diagonal blocks in L
    """
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
