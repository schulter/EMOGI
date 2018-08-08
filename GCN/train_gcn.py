import argparse
import os
from datetime import datetime
import tensorflow as tf
import gcn.utils
import utils
from my_gcn import MYGCN
#import interpretation

from scipy.sparse import lil_matrix
import scipy.sparse as sp
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
import pandas as pd
import seaborn
from tensorflow.contrib.tensorboard.plugins import projector
import pickle as pkl
import networkx as nx


def interpret_results(model_dir):
    print("Running feature interpretation for {}".format(model_dir))
    genes = ["CEBPB", "CHD1", "CHD3", "CHD4", "TP53", "PADI4", "RBL2",
             "BRCA1", "BRCA2", "NOTCH2", "NOTCH1", "MYOC", "ZNF24", "SIM1",
             "HSP90AA1", "ARNT"]
    interpretation.interpretation(
        model_dir, genes, os.path.join(model_dir, 'lrp'), True)


def write_hyper_params(args, input_file, file_name):
    with open(file_name, 'w') as f:
        for arg in vars(args):
            f.write('{}\t{}\n'.format(arg, getattr(args, arg)))
        f.write('{}\n'.format(input_file))
    print("Hyper-Parameters saved to {}".format(file_name))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train GCN model and save to file')
    parser.add_argument('-e', '--epochs', help='Number of Epochs',
                        dest='epochs',
                        default=150,
                        type=int
                        )
    parser.add_argument('-lr', '--learningrate', help='Learning Rate',
                        dest='lr',
                        default=.1,
                        type=float
                        )
    parser.add_argument('-s', '--support', help='Neighborhood Size in Convolutions',
                        dest='support',
                        default=1,
                        type=int
                        )
    parser.add_argument('-hd', '--hidden_dims',
                        help='Hidden Dimensions (number of filters per layer. Also determines the number of hidden layers.',
                        nargs='+',
                        dest='hidden_dims',
                        required=True)
    parser.add_argument('-lm', '--loss_mul',
                        help='Number of times, false negatives are weighted higher than false positives',
                        dest='loss_mul',
                        default=1,
                        type=float
                        )
    parser.add_argument('-wd', '--weight_decay', help='Weight Decay',
                        dest='decay',
                        default=5e-4,
                        type=float
                        )
    parser.add_argument('-do', '--dropout', help='Dropout Percentage',
                        dest='dropout',
                        default=.5,
                        type=float
                        )
    parser.add_argument('-d', '--data', help='Path to HDF5 container with data',
                        dest='data',
                        type=str
                        )
    args = parser.parse_args()
    return args


def fits_on_gpu(adj, features, hidden_dims, support):
    """Determines if training should be done on the GPU or CPU.
    """
    total_size = 0
    cur_dim = features[2][1]
    n = features[2][0]
    sp_adj = gcn.utils.preprocess_adj(adj)
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


def predict(sess, model, features, support, labels, mask, placeholders):
    feed_dict_pred = gcn.utils.construct_feed_dict(
        features, support, labels, mask, placeholders)
    pred = sess.run(model.predict(), feed_dict=feed_dict_pred)
    return pred

class EarlyStoppingMonitor():
    def __init__(self, patience):
        self.patience = patience
        self.epochs_without_improvement = 0
        self.best_loss = np.inf
    
    def should_stop(self, loss):
        if self.best_loss <= loss: # no improvement
            self.epochs_without_improvement += 1
            print (loss, self.best_loss)
        else: # improvement
            self.epochs_without_improvement = 0
            self.best_loss = loss

        # shall be stop?
        print ("epochs without improvement: {}".format(self.epochs_without_improvement))
        if self.epochs_without_improvement >= self.patience:
            return True
        else:
            return False


if __name__ == "__main__":
    args = parse_args()
    if not args.data.endswith('.h5'):
        print("Data is not hdf5 container. Exit now.")
        sys.exit(-1)

    input_data_path = args.data
    data = utils.load_hdf_data(input_data_path, feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names = data
    print("Read data from: {}".format(input_data_path))
    num_nodes = adj.shape[0]
    num_feat = features.shape[1]
    if num_feat > 1:
        features = utils.preprocess_features(lil_matrix(features))
    else:
        print("Not row-normalizing features because feature dim is {}".format(num_feat))
        features = gcn.utils.sparse_to_tuple(lil_matrix(features))

    # preprocess adjacency matrix and account for larger support
    poly_support = args.support
    if poly_support > 0:
        support = gcn.utils.chebyshev_polynomials(adj, poly_support)
        num_supports = 1 + poly_support
    else:  # support is 0, don't use the network
        support = [sp.eye(adj.shape[0])]
        num_supports = 1

    # create placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=features[2]),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)
    }
    hidden_dims = [int(x) for x in args.hidden_dims]

    with tf.Session() as sess:
        model = MYGCN(placeholders=placeholders,
                      input_dim=features[2][1],
                      learning_rate=args.lr,
                      weight_decay=args.decay,
                      num_hidden_layers=len(args.hidden_dims),
                      hidden_dims=hidden_dims,
                      pos_loss_multiplier=args.loss_mul,
                      logging=True)

        performance_ops = model.get_performance_metrics()
        running_avg_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                             scope="evaluation")
        metric_reset_op = tf.variables_initializer(var_list=running_avg_vars)
        early_stopping_mon = EarlyStoppingMonitor(patience=10)
        # create model directory for saving
        root_dir = '../data/GCN/training'
        if not os.path.isdir(root_dir):  # in case training root doesn't exist
            os.mkdir(root_dir)
            print("Created Training Subdir")
        date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.mkdir(os.path.join(root_dir, date_string))
        save_path = os.path.join(root_dir, date_string)

        # initialize writers for TF logs
        merged = tf.summary.merge_all()
        val_summary = tf.summary.merge_all(scope='evaluation')
        config = projector.ProjectorConfig()
        train_writer = tf.summary.FileWriter(os.path.join(save_path, 'train'))
        test_writer = tf.summary.FileWriter(os.path.join(save_path, 'test'))

        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        for epoch in range(args.epochs):
            feed_dict = gcn.utils.construct_feed_dict(features, support, y_train,
                                                      train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: args.dropout})
            # Training step
            #_ = sess.run(metric_reset_op)
            _ = sess.run(model.opt_op, feed_dict=feed_dict)
            train_loss, train_acc, train_aupr, train_auroc = sess.run(performance_ops,
                                                                      feed_dict=feed_dict)
            s = sess.run(merged, feed_dict=feed_dict)
            train_writer.add_summary(s, epoch)
            train_writer.flush()

            # Print validation accuracy once in a while
            if epoch % 10 == 0 or epoch-1 == args.epochs:
                d = gcn.utils.construct_feed_dict(features, support, y_test,
                                                  test_mask, placeholders)
                sess.run(metric_reset_op)
                val_loss, val_acc, val_aupr, val_auroc = sess.run(performance_ops,
                                                                  feed_dict=d)
                s = sess.run(merged, feed_dict=d)
                test_writer.add_summary(s, epoch)
                test_writer.flush()
                print("Epoch:", '%04d' % (epoch + 1),
                      "Test Loss=", "{:.5f}".format(val_loss),
                      "Test Acc=", "{:.5f}".format(val_acc),
                      "Test AUROC={:.5f}".format(val_auroc),
                      "Test AUPR: {:.5f}".format(val_aupr))
                if early_stopping_mon.should_stop(val_loss):
                    print ("Early Stopping")
                    break
            print("Epoch:", '%04d' % (epoch + 1),
                    "Train Loss=", "{:.5f}".format(train_loss),
                    "Train Acc=", "{:.5f}".format(train_acc),
                    "Train AUROC={:.5f}".format(train_auroc),
                    "Train AUPR: {:.5f}".format(train_aupr))
        print("Optimization Finished!")

        # save model
        model_save_path = os.path.join(save_path, 'model.ckpt')
        print("Save model to {}".format(model_save_path))
        path = model.save(model_save_path, sess=sess)

        # Testing
        d = utils.construct_feed_dict(features, support, y_test, test_mask, placeholders)
        test_performance = sess.run(performance_ops, feed_dict=d)
        print("Test set results:", "loss=", "{:.5f}".format(test_performance[0]),
              "accuracy=", "{:.5f}".format(
                  test_performance[1]), "aupr=", "{:.5f}".format(test_performance[2]),
              "auroc=", "{:.5f}".format(test_performance[3]))

        # predict all nodes (result from algorithm)
        predictions = predict(sess, model, features, support, y_test,
                              test_mask, placeholders)

    # save predictions
    with open(os.path.join(save_path, 'predictions.tsv'), 'w') as f:
        f.write('ID\tName\tProb_pos\n')
        for pred_idx in range(predictions.shape[0]):
            f.write('{}\t{}\t{}\n'.format(node_names[pred_idx, 0],
                                            node_names[pred_idx, 1],
                                            predictions[pred_idx, 0])
                    )
    # save hyper Parameters and plot
    write_hyper_params(args, input_data_path, os.path.join(
        save_path, 'hyper_params.txt'))
    utils.plot_roc_pr_curves(
        predictions[test_mask == 1], y_test[test_mask == 1], save_path)
    # interpret_results(save_path)
