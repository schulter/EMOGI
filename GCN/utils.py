import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, h5py, os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_cora():
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../../gcn/gcn/data/ind.cora.{}".format(names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = gcn.utils.parse_index_file(
        "../../gcn/gcn/data/ind.cora.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = gcn.utils.sample_mask(idx_train, labels.shape[0])
    val_mask = gcn.utils.sample_mask(idx_val, labels.shape[0])
    test_mask = gcn.utils.sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def plot_roc_pr_curves(y_score, y_true, model_dir):
    # define y_true and y_score
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    # plot ROC curve
    fig = plt.figure(figsize=(14, 8))
    plt.plot(fpr, tpr, lw=3, label='AUC = {0:.2f}'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='gray', lw=3,
             linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Prediction on Train and Test')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(model_dir, 'roc_curve.png'))

    # plot PR-Curve
    pr, rec, thresholds = precision_recall_curve(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    fig = plt.figure(figsize=(14, 8))
    plt.plot(rec, pr, lw=3, label='GCN (AUPR = {0:.2f})'.format(aupr))
    random_y = y_true.sum() / (y_true.sum() + y_true.shape[0] - y_true.sum())
    plt.plot([0, 1], [random_y, random_y], color='gray', lw=3, linestyle='--',
             label='Random')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve on Train and Test')
    plt.legend()
    fig.savefig(os.path.join(model_dir, 'prec_recall.png'))



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features, sparse=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if sparse:
        return sparse_to_tuple(features)
    else:
        return features.todense()


def normalize_adj(adj, sparse=True):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    if sparse:
        return res.tocoo()
    else:
        return res.todense()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def load_hdf_data(path, network_name='network', feature_name='features'):
    with h5py.File(path, 'r') as f:
        network = f[network_name][:]
        features = f[feature_name][:]
        node_names = f['gene_names'][:]
        y_train = f['y_train'][:]
        y_test = f['y_test'][:]
        if 'y_val' in f:
            y_val = f['y_val'][:]
        else:
            y_val = None
        train_mask = f['mask_train'][:]
        test_mask = f['mask_test'][:]
        if 'mask_val' in f:
            val_mask = f['mask_val'][:]
        else:
            val_mask = None
    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names


def chebyshev_polynomials(adj, k, sparse=True, subtract_support=True):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj, sparse=sparse)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    if sparse:
        t_k.append(sp.eye(adj.shape[0]))
    else:
        t_k.append(np.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    if sparse:
        if subtract_support:
            t_k = subtract_lower_support(t_k)
        return sparse_to_tuple(t_k)
    else:
        if subtract_support:
            return subtract_lower_support(t_k)
        else:
            return t_k

def subtract_lower_support(polys):
    for i in range(1, len(polys)):
        for j in range(0, i):
            polys[i][np.abs(polys[j]) > 0.0001] = 0
    return polys