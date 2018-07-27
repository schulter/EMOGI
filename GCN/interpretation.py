import tensorflow as tf
import os, sys
import h5py
import gcn.utils
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import numpy as np
import math
from deepexplain.tensorflow import DeepExplain
sys.path.append(os.path.abspath('../GCN'))
from my_gcn import MYGCN
import utils


def str_to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def load_hyper_params(model_dir):
    file_name = os.path.join(model_dir, 'hyper_params.txt')
    input_file = None
    with open(file_name, 'r') as f:
        args = {}
        for line in f.readlines():
            if '\t' in line:
                key, value = line.split('\t')
                if value.startswith('['): # list of hidden dimensions
                    f = lambda x: "".join(c for c in x if c not in ['\"', '\'', ' ', '\n', '[', ']']) 
                    l = [int(f(i)) for i in value.split(',')]
                    args[key.strip()] = l
                else:
                    args[key.strip()] = str_to_num(value.strip())
            else:
                input_file = line.strip()
    return args, input_file


def load_hdf_data(path, network_name='network', feature_name='features'):
    with h5py.File(path, 'r') as f:
        network = f[network_name][:]
        features = f[feature_name][:]
        node_names = f['gene_names'][:]
        y_train = f['y_train'][:]
        train_mask = f['mask_train'][:]
        if 'feature_names' in f:
            feature_names = f['feature_names'][:]
        else:
            feature_names = None
        # y_test = f['y_test'][:]
        # test_mask = f['mask_test'][:]
        # y_val = f['y_val'][:]
        # val_mask = f['mask_val'][:]
    return network, features, y_train, train_mask, node_names, feature_names


def get_direct_neighbors(adj, node):
    neighbors = []
    for idx, val in enumerate(adj[node,:]):
        if math.isclose(val, 1):
            neighbors.append(idx)
    return neighbors


def get_top_neighbors(idx_gene, adj, attributions):
    edge_list = []
    nodes_attr = {}
    sources = [idx_gene]
    for support in range(2, len(attributions)):
        for source in sources:
            next_neighbors = get_direct_neighbors(adj, source)
            for next_node in next_neighbors:
                if not (next_node, source) in edge_list:
                    edge_list.append((source, next_node))
                    if not next_node in nodes_attr:
                        val = (attributions[support][idx_gene, next_node] + attributions[support][next_node, idx_gene])/2
                        nodes_attr[next_node] = val
        sources = next_neighbors
    return edge_list, nodes_attr


def save_edge_list(edge_list, nodes_attr, gene, node_names, out_dir):
    with open("{}{}.edgelist".format(out_dir, gene), "wt") as out_handle:
        out_handle.write("SOURCE\tTARGET\tLRP_ATTR_TARGET\tLABEL_TARGET\n")
        for edge in edge_list:
            out_handle.write("{}\t{}\t{}\t{}\n".format(*edge, nodes_attr[edge[1]], node_names[edge[1]]))


def _hide_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def save_plots(feature_names, idx_gene, features, feature_attr, node_names, most_important, least_important, out_dir):
    fig, ax = plt.subplots(nrows = 3, ncols = 1, figsize = (12, 6))
    # original feature plot
    x = np.arange(len(feature_names))
    ax[0].set_title(node_names[idx_gene])
    ax[0].bar(x, features[idx_gene,:].tolist()[0], color="#67a9cf")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(feature_names)
    ax[0].set_ylabel("feature value")
    # LRP attributions for each feature
    ax[1].bar(x, feature_attr[idx_gene,:], color="#67a9cf")
    ax[1].set_xticklabels(feature_names)
    ax[1].set_xticks(x)
    ax[1].set_ylabel("LRP attribution")
    # most/least important neighbors
    neighbors = most_important+least_important[::-1]
    x = np.arange(len(neighbors))
    ax[2].bar(x, [i[1] for i in neighbors], color="#5ab4ac")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([i[0] for i in neighbors])
    ax[2].set_ylabel("LRP attribution")
    for i in range(len(most_important),len(neighbors)):
        ax[2].patches[i].set_facecolor("#d8b365")
    # finalize
    _hide_top_right(ax[0])
    _hide_top_right(ax[1])
    _hide_top_right(ax[2])
    for axis in ax:
        for tick in axis.get_xticklabels():
            tick.set_rotation(90)
    plt.tight_layout()
    fig.savefig("{}{}.pdf".format(out_dir, node_names[idx_gene]))
    fig.clf()
    plt.close('all')


def compute_and_summarize_LRP(de, model, idx_gene, placeholders, features, support, adj, node_names, feature_names,
                              out_dir, save_edge_lists):
    mask_gene = np.zeros((features.shape[0],1))
    mask_gene[idx_gene] = 1
    attributions = de.explain(method="elrp",
                              T=model.outputs * mask_gene,
                              X=[placeholders['features'], *placeholders["support"]],
                              xs=[features, *support])
    edge_list, nodes_attr = get_top_neighbors(idx_gene, adj, attributions)
    if save_edge_lists:
        save_edge_list(edge_list, nodes_attr, node_names[idx_gene], node_names, out_dir)
    nodes_sorted = sorted(nodes_attr.items(), key=lambda x: x[1])
    nodes_sorted = [(node_names[idx], attr) for idx, attr in nodes_sorted]
    most_important = nodes_sorted[-15:][::-1]
    least_important = nodes_sorted[:15]
    # plots
    f_names = feature_names if not feature_names is None else np.arange(features.shape[1])
    save_plots(f_names, idx_gene, features, attributions[0],
               node_names, most_important, least_important, out_dir)
    return most_important, least_important


def interpretation(model_dir, genes, out_dir, save_edge_lists=False):
    if out_dir[-1] != "/":
        out_dir += "/"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    args, data_file = load_hyper_params(model_dir)
    print(data_file)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_dir)

    data = load_hdf_data(data_file, feature_name='features')
    adj, features, y_train, train_mask, node_names, feature_names = data
    node_names = [x[1] for x in node_names]
    features = utils.preprocess_features(lil_matrix(features), sparse=False)

    if args["support"] > 0:
        support = utils.chebyshev_polynomials(adj, args["support"], sparse=False)
        num_supports = 1 + args["support"]
    else:
        support = [np.eye(adj.shape[0])]
        num_supports = 1
    
    placeholders = {
        'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.placeholder(tf.float32, shape=features.shape),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)
    }

    results_genes = {}
    with tf.Session() as sess:
        with DeepExplain(session=sess) as de:
            model = MYGCN(placeholders=placeholders,
                          input_dim=features.shape[1],
                          learning_rate=args['lr'],
                          weight_decay=args['decay'],
                          num_hidden_layers=len(args['hidden_dims']),
                          hidden_dims=args['hidden_dims'],
                          pos_loss_multiplier=args['loss_mul'],
                          logging=False, sparse_network=False)
            model.load(ckpt.model_checkpoint_path, sess)

            for gene in genes:
                try:
                    idx_gene = node_names.index(gene)
                except:
                    print("Warning: '{}' not found in label list".format(gene))
                    continue
                print("Now: {}".format(gene))
                highest_attr, lowest_attr = compute_and_summarize_LRP(de, model, idx_gene, placeholders,
                                                                      features, support, adj, node_names,
                                                                      feature_names, out_dir, save_edge_lists)
                results_genes[gene] = (highest_attr, lowest_attr)


def main():
    # '../data/GCN/training/pancancer_meth_450k_1000bpprom/'
    # '../data/GCN/training/pancancer_tcgage/'
    # '../data/GCN/training/simulation_LRP_subsupport_all/'
    interpretation(model_dir = '../data/GCN/training/pancancer_multiomics_methpromonly1000_tcgage',
                   genes = ["NCK1", "ITGAX", "PAG1", "SH2B2", "CEBPB", "CHD1", "CHD3", "CHD4", "TP53", "PADI4", "RBL2", "BRCA1", "BRCA2", "NOTCH2", "NOTCH1",
                            "MYOC", "ZNF24", "SIM1", "HSP90AA1", "ARNT"],
                   out_dir = "tests_ge/",
                   save_edge_lists = True)


if __name__ == "__main__":
    main()