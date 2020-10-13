import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def str_to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _plot_hide_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def colorize_by_omics(axes, feature_names):
    # look at feature names and colorize the bars in the axes object
    for idx, feat in enumerate(feature_names):
        if feat.startswith("MF:"):
            col = "#800000"
        elif feat.startswith("METH:"):
            col = "#00008b"
        elif feat.startswith("GE:"):
            col = "#006400"
        else:
            col = "#001f3f"
        axes.patches[idx].set_facecolor(col)
        axes.get_xticklabels()[idx].set_color(col)


def lrp_heatmap_plot(fig, outer_grid, x, xlabels, title=None):
    x = np.abs(x)
    if not len(x.shape) == 2: # we have only 1D input, make it 2D
        x = x.reshape(16, -1, order='F')
    inner = gridspec.GridSpecFromSubplotSpec(x.shape[1], 1, hspace=0, subplot_spec=outer_grid)
    if x.shape[1] == 3:
        omics = ['Mutation', 'Methylation', 'Expression']
        cmaps = [sns.color_palette("Reds", n_colors=50),
             sns.color_palette("Blues", n_colors=50),
             sns.color_palette("Greens", n_colors=50)]
    elif x.shape[1] == 4:
        omics = ['Mutation', 'Copy Num. Change', 'Methylation', 'Expression']
        cmaps = [sns.color_palette("Reds", n_colors=50),
             sns.color_palette("Purples", n_colors=50),
             sns.color_palette("Blues", n_colors=50),
             sns.color_palette("Greens", n_colors=50)] # for CNAs
        x[:,[1, 3]] = x[:,[3, 1]]
        x[:,[2, 3]] = x[:,[3, 2]] # order by SNV, CNA, Meth, Expr.
    else:
        print ("Features not understood. Maybe you have a patient-wise model. Try with --bars True.")
    vmax = x.max()
    vmin = x.min()
    print ("Min: {}\tMax: {}".format(vmin, vmax))
    for c in range(x.shape[1]):
        ax = plt.Subplot(fig, inner[c])
        xticklabels = False
        if c == x.shape[1] - 1: # last omics
            xticklabels = [i.split(':')[1] for i in xlabels if i.startswith('MF:')]
            #xticklabels = xlabels
        sns.heatmap(x[:, c].reshape(1, -1), ax=ax, xticklabels=xticklabels,
                    cbar=False, cmap=cmaps[c],
                    cbar_kws={'use_gridspec': False, 'orientation': 'vertical'},
                    vmax=vmax, vmin=vmin)
        ax.set_yticklabels([omics[c]], rotation=0, fontsize=10)
        if c == x.shape[1] - 1:
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=10)
        if not title is None and c == 0:
            ax.set_title(title, fontsize=16)
        fig.add_subplot(ax)
    plt.subplots_adjust(bottom=0.05, hspace=0.05, wspace=0)



def lrp_barplot(fig, outer_grid, x, xlabels, std=None, y_name=None, title=None):
    ax = plt.Subplot(fig, outer_grid)
    ax.bar(np.arange(len(xlabels)), x, yerr=std, tick_label=xlabels)
    if not title is None:
        ax.set_title(title)
    if not y_name is None:
        ax.set_ylabel(y_name)
    colorize_by_omics(ax, xlabels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    fig.add_subplot(ax)



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


def plot_roc_pr_curves(y_score, y_true, model_dir):
    # define y_true and y_score
    fpr, tpr, _ = roc_curve(y_true, y_score)
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
    pr, rec, _ = precision_recall_curve(y_true, y_score)
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
    #feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict



def get_neighborhood_support(adj, k, sparse=True):
    """Calculate the support matrices for up to order k.
    
    This method calculates the neighborhoods up to order k from
    an adjacency matrix adj.
    From those, all lower support nodes will be subtracted, leaving
    only nodes that were unreachable from a given node with k-1 hops
    but are reachable with k hops.
    """
    I = np.eye(adj.shape[0])
    support_matrices = [I, adj]
    sum_so_far = adj
    for i in range(1, k):
        S = ((np.linalg.matrix_power(adj, i) > 0) - sum_so_far > 0).astype(np.int32)
        sum_so_far = sum_so_far + S
        support_matrices.append(S)
    
    if sparse:
        support_matrices = [sp.coo_matrix(s) for s in support_matrices]
        return sparse_to_tuple(support_matrices)
    else:
        return support_matrices

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

    for _ in range(2, k+1):
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



def fits_on_gpu(adj, features, hidden_dims, support):
    """Determines if training should be done on the GPU or CPU.
    """
    total_size = 0
    cur_dim = features[2][1]
    n = features[2][0]
    sp_adj = preprocess_adj(adj)
    adj_size = np.prod(sp_adj[0].shape) + np.prod(sp_adj[1].shape)
    print(adj_size, n)

    for layer in range(len(hidden_dims)):
        H_s = n * cur_dim
        W_s = cur_dim * hidden_dims[layer]
        total_size += (adj_size + H_s + W_s)*support
        cur_dim = hidden_dims[layer]
        print(H_s, W_s, total_size, cur_dim)
    total_size *= 4  # assume 32 bits (4 bytes) per number

    print(total_size, total_size < 11*1024*1024*1024)
    return total_size < 11*1024*1024*1024  # 12 GB memory (only take 11)

def get_support_matrices(adj, poly_support):
    if poly_support > 0:
        support = chebyshev_polynomials(adj, poly_support)
        #support = get_neighborhood_support(adj, poly_support)
        num_supports = 1 + poly_support
    else:  # support is 0, don't use the network
        support = [sp.eye(adj.shape[0])]
        num_supports = 1
    return support, num_supports

class EarlyStoppingMonitor():
    def __init__(self, model, sess, path, patience):
        self.target_model = model
        self.tfsession = sess
        self.model_save_path = path
        self.patience = patience
        self.epochs_without_improvement = 0
        self.best_score = np.inf

    def should_stop(self, score):
        if self.best_score <= score: # no improvement
            self.epochs_without_improvement += 1
            print (score, self.best_score)
        else: # improvement
            self.epochs_without_improvement = 0
            self.best_score = score
            self.target_model.save(self.model_save_path, self.tfsession)

        # shall be stop?
        print ("epochs without improvement: {}".format(self.epochs_without_improvement))
        if self.epochs_without_improvement >= self.patience:
            return True
        else:
            return False