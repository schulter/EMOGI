import argparse
import os, h5py
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import gcn.utils
from my_gcn import MYGCN
from scipy.sparse import csr_matrix, lil_matrix
from gcn.models import GCN
import time

from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import average_precision_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_hdf_data(path):
    with h5py.File(path, 'r') as f:
        network = f['network'][:]
        features = f['features'][:]
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

def masked_aupr_score(y_true, y_score, mask):
    y_true_masked = y_true[:, 0][mask > 0.5]
    y_score_masked = y_score[:, 0][mask > 0.5]
    return average_precision_score(y_true=y_true_masked, y_score=y_score_masked)

def evaluate(model, session, features, support, labels, mask, placeholders):
    feed_dict_val = gcn.utils.construct_feed_dict(features, support, labels, mask, placeholders)
    loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return loss, acc

def predict(model, session, features, support, labels, mask, placeholders):
    feed_dict_pred = gcn.utils.construct_feed_dict(features, support, labels, mask, placeholders)
    pred = sess.run(model.predict(), feed_dict=feed_dict_pred)
    return pred

def fits_on_gpu(adj, features, hidden_dims, support):
    """Determines if training should be done on the GPU or CPU.
    """
    total_size = 0
    cur_dim = features[2][1]
    n = features[2][0]
    sp_adj = gcn.utils.preprocess_adj(adj)
    adj_size = np.prod(sp_adj[0].shape) + np.prod(sp_adj[1].shape)
    print (adj_size, n)

    for layer in range(len(hidden_dims)):
        H_s = n * cur_dim
        W_s = cur_dim * hidden_dims[layer]
        total_size += (adj_size + H_s + W_s)*support
        cur_dim = hidden_dims[layer]
    total_size *= 4 # assume 32 bits (4 bytes) per number

    return total_size < 11*1024*1024*1024 # 12 GB memory (only take 11)
    

def cv_split(y, mask, val_size):
    """Split mask and targets into train and validation sets (stratified).

    This method contructs mask and targets for training and validation
    from the complete mask and targets. The proportion of nodes used
    for validation is determined by `val_size`.
    The split will be stratified and the returned arrays have the
    same dimensions as the input arrays.

    Parameters:
    ----------
    y:                  The targets for all nodes.
    mask:               The mask for the known nodes
    val_size:           The proportion (or absolute size) of nodes
                        used for validation
    
    Returns:
    Four arrays of length of the input arrays, namely train targets,
    train mask, validation targets and validation mask.
    """
    assert (y.shape[0] == mask.shape[0])
    mask_idx = np.where(mask == 1)[0]
    train_idx, val_idx = train_test_split(mask_idx, test_size=val_size,
                                          stratify=y[mask==1, 0])
    # build the train/validation masks
    m_t = np.zeros_like(mask)
    m_t[train_idx] = 1
    m_v = np.zeros_like(mask)
    m_v[val_idx] = 1

    # build the train/validation targets
    y_t = np.zeros_like(y)
    y_t[train_idx] = y[train_idx] # all train nodes get train targets
    y_v = np.zeros_like(y)
    y_v[val_idx] = y[val_idx] # same for validation

    return y_t, m_t, y_v, m_v

def run_cv(model, sess, features, num_runs, params, placeholders, support, y, mask):
    """Run one parameter setting with CV and evaluate on validation data.
    """
    # where the results go
    accs = []
    losses = []
    auprs = []
    num_preds = []

    for cv_run in range(num_runs):
        # select some training genes randomly
        size = 1 / float(num_runs) # size for validation (1/CV runs)
        y_train, train_mask, y_val, val_mask = cv_split(y, mask, size)
        merged = tf.summary.merge_all()
        sess.run(tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()))
        for epoch in range(params['epochs']):
            feed_dict = gcn.utils.construct_feed_dict(features, support, y_train,
                                                      train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: params['dropout']})
            outs = sess.run([model.opt_op, merged],
                            feed_dict=feed_dict)
        # Testing
        val_loss, val_acc = evaluate(model, sess, features, support,
                                       y_val, val_mask, placeholders)
        predictions = predict(model, sess, features, support,
                              y_val, val_mask, placeholders)
        num_pos_pred = (predictions[:, 0] > .5).sum()
        num_preds.append(num_pos_pred)
        accs.append(val_acc)
        losses.append(val_loss)
        aupr = masked_aupr_score(y_test, predictions, test_mask)
        auprs.append(aupr)
    print ("Test AUPR: {}".format(np.mean(auprs)))
    return accs, losses, num_preds, auprs


def run_model(session, params, adj, features, y_train, y_test, train_mask, test_mask):
    poly_support = params['support']
    if poly_support > 1:
        support = gcn.utils.chebyshev_polynomials(adj, poly_support)
        num_supports = 1 + poly_support
    else:
        support = [gcn.utils.preprocess_adj(adj)]
        num_supports = 1
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    model = MYGCN(placeholders=placeholders,
                  input_dim=features[2][1],
                  learning_rate=params['learningrate'],
                  weight_decay=params['weight_decay'],
                  num_hidden_layers=len(params['hidden_dims']),
                  hidden_dims=params['hidden_dims'],
                  pos_loss_multiplier=params['loss_mul'],
                  logging=True)
    return run_cv(model, sess, features, 5, params, placeholders,
                  support, y_train, train_mask)

if __name__ == "__main__":
    print ("Loading Data...")
    cv_runs = 5
    data = load_hdf_data('../data/cancer/hotnet_iref_vec_input_unbalanced.h5')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names = data
    num_nodes = adj.shape[0]
    num_feat = features.shape[1]
    if num_feat > 1:
        features = gcn.utils.preprocess_features(lil_matrix(features))
    else:
        print ("Not row-normalizing features because feature dim is {}".format(num_feat))
        features = gcn.utils.sparse_to_tuple(lil_matrix(features))

    params = {'support':[1, 2],
              'dropout':[.1],
              'hidden_dims': [[20, 40], [20, 40, 80], [80, 40, 20], [5], [20, 40, 40, 20],
                             [100], [100, 200], [5, 40, 5]],
              'loss_mul': [1, 50, 175, 250],
              'learningrate':[0.1, .01, .0005],
              'epochs':[700],
              'weight_decay':[5e-4, 5e-2]
              }

    #params = {'support':[2], 'dropout':[.1], 'hidden_dims':[[40, 80]], 'loss_mul':[175], 'learningrate':[.01], 'epochs':[500], 'weight_decay':[5e-4]}
 
    num_of_settings = len(list(ParameterGrid(params)))
    print ("Grid Search: Trying {} different parameter settings...".format(num_of_settings))
    param_num = 1
    # create session, train and save afterwards
    performances = []
    for param_set in list(ParameterGrid(params)):
        with tf.Session() as sess:
            accs, losses, numpreds, auprs = run_model(sess, param_set, adj,
                                                      features, y_train, y_test,
                                                      train_mask, test_mask)
        performances.append((accs, losses, numpreds, auprs, param_set))
        print ("[{} out of {} combinations]: {}".format(param_num, num_of_settings, param_set))
        param_num += 1
        tf.reset_default_graph()
    # write results from gridsearch to file
    out_name = '../data/gridsearch/gridsearchcv_results_cancer_vec_unbalanced.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(performances, f)
