import h5py
import os
import gcn
import numpy as np
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
from datetime import datetime

def load_cora():
    """Load the Cora data set from the orignal GCN paper."""
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

def load_hdf_data(path, network_name='network', feature_name='features'):
    """Load a GCN input HDF5 container and return its content.

    This funtion reads an already preprocessed data set containing all the
    data needed for training a GCN model in a medical application.
    It extracts a network, features for all of the nodes, the names of the
    nodes (genes) and training, testing and validation splits.

    Paramters:
    ---------
    path:               Path to the container
    network_name:       Sometimes, there might be different networks in the
                        same HDF5 container. This name specifies one of those.
                        Default is: 'network'
    feature_name:       The name of the features of the nodes. Default is: 'features'

    Returns:
    A tuple with all of the data in the order: network, features, y_train, y_val,
    y_test, train_mask, val_mask, test_mask, node names.
    """
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

def write_hyper_params(args, input_file, file_name):
    with open(file_name, 'w') as f:
        for arg in vars(args):
            f.write('{}\t{}\n'.format(arg, getattr(args, arg)))
        f.write('{}\n'.format(input_file))
    print("Hyper-Parameters saved to {}".format(file_name))


def str_to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def load_hyper_params(model_dir):
    """Loads a set of hyper parameters from disk.

    Loads a set of hyper params from disk that were stored before
    by the 'write_hyper_params' function.

    Parameters:
    ----------
    model_dir:          The directory in which the model and the hyper
                        parameter text file are located.
    Returns:
    A dictionary containing the hyper parameters.
    """
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
    print ("Hyper-Parameters read from {}".format(file_name))
    return args, input_file



def create_model_dir():
    root_dir = '../data/GCN/training'
    if not os.path.isdir(root_dir):  # in case training root doesn't exist
        os.mkdir(root_dir)
        print("Created Training Subdir")
    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(os.path.join(root_dir, date_string))
    model_path = os.path.join(root_dir, date_string)
    return model_path

def save_predictions(output_dir, node_names, predictions):
    with open(os.path.join(output_dir, 'predictions.tsv'), 'w') as f:
        f.write('ID\tName\tProb_pos\n')
        for pred_idx in range(predictions.shape[0]):
            f.write('{}\t{}\t{}\n'.format(node_names[pred_idx, 0],
                                            node_names[pred_idx, 1],
                                            predictions[pred_idx, 0])
            )


def write_train_test_sets(out_dir, y_train, y_test, train_mask, test_mask):
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(out_dir, 'train_mask.npy'), train_mask)
    np.save(os.path.join(out_dir, 'test_mask.npy'), test_mask)


def read_train_test_sets(dir):
    y_train = np.load(os.path.join(dir, 'y_train.npy'))
    y_test = np.load(os.path.join(dir, 'y_test.npy'))
    train_mask = np.load(os.path.join(dir, 'train_mask.npy'))
    test_mask = np.load(os.path.join(dir, 'test_mask.npy'))
    return y_train, y_test, train_mask, test_mask