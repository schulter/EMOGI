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
from sklearn.model_selection import ParameterGrid
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

def run_cv(model, sess, num_runs, params, placeholders, support):
    accs = []
    losses = []
    auprs = []
    num_preds = []
    for cv_run in range(num_runs):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        cost_val = []
        for epoch in range(params['epochs']):
            t = time.time()
            feed_dict = gcn.utils.construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: params['dropout']})
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        # Testing
        test_cost, test_acc = evaluate(model, sess, features, support, y_test, test_mask, placeholders)
        predictions = predict(model, sess, features, support, y_test, test_mask, placeholders)
        num_pos_pred = (predictions[:, 0] > .5).sum()
        num_preds.append(num_pos_pred)
        accs.append(test_acc)
        losses.append(test_cost)
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
                  logging=False)
    return run_cv(model, sess, 5, params, placeholders, support)

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
              'hidden_dims': [[20, 40], [20, 40, 80], [80, 40, 20, 40]],
              'loss_mul': [1, 50, 200],
              'learningrate':[0.1, .01, .001, .0005],
              'epochs':[500],
              'weight_decay':[5e-4, 5e-3, 5e-2]
              }

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
    with open('../data/gridsearch/gridsearch_results_unbalanced.pkl', 'wb') as f:
        pickle.dump(performances, f)
