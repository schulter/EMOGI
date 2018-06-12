import argparse
import os
import h5py
from datetime import datetime
import tensorflow as tf
import gcn.utils
from my_gcn import MYGCN
from scipy.sparse import lil_matrix
import scipy.sparse as sp
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

import numpy as np
import pandas as pd
import seaborn
from tensorflow.contrib.tensorboard.plugins import projector
import pickle as pkl
import networkx as nx


def bestSplit(x): return (round(math.sqrt(x)),
                          math.ceil(x / round(math.sqrt(x))))


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


def get_color_dataframe(node_names, predictions, y_train, y_test):
    labels_df = pd.DataFrame(node_names, index=node_names[:, 0], columns=[
                             'ID', 'name']).drop('ID', axis=1)
    labels_df['label'] = (y_train[:, 0] | y_test[:, 0])
    labels_df['train_label'] = y_train[:, 0]
    labels_df['test_label'] = y_test[:, 0]
    labels_df['prediction'] = predictions[:, 0] >= 0.5
    labels_df['color'] = 'gray'
    labels_df.loc[labels_df.prediction == 1, 'color'] = 'blue'
    labels_df.loc[labels_df.train_label == 1, 'color'] = 'green'
    labels_df.loc[labels_df.test_label == 1, 'color'] = 'red'
    return labels_df


def plot_weights(sess, model, dir_name):
    for var in model.vars:
        layer_num = int(var.split('/')[1].strip().split('_')[1])
        # read weight matrix from TF
        weight_mat = model.vars[var].eval(session=sess)
        feature_size, embed_size = weight_mat.shape

        # assign feature names (either experiments or embeddings from prev layer)
        if layer_num == 1:
            feature_names = ['Pam3T16', 'Pam3T8', 'Pam3T16.1',
                             'Pam3T8.1', 'Pam3T16.2', 'Pam3T8.2',
                             'ControlT8', 'ControlT16', 'ControlT8.1',
                             'ControlT16.1', 'ControlT8.2', 'ControlT16.2',
                             'gfpmT8', 'gfpmT16', 'gfpmT8.1', 'gfpmT16.1',
                             'gfpmT8.2', 'gfpmT16.2', 'gfppT8', 'gfppT16',
                             'gfppT8.1', 'gfppT16.1', 'gfppT8.2', 'gfppT16.2']
        else:
            feature_names = ['filter_{}'.format(
                i) for i in range(feature_size)]

        # plot figure itself
        fig = plt.figure(figsize=(30, 20))
        plt.gcf().subplots_adjust(left=0.2)
        num_rows, num_cols = bestSplit(embed_size)
        for i in range(embed_size):
            if i > 1:
                ax = plt.subplot(num_rows, num_cols, i+1, sharey=ax, sharex=ax)
            else:
                ax = plt.subplot(num_rows, num_cols, i+1)
            plt.barh(np.arange(0, feature_size), weight_mat[:, i])
            plt.yticks(np.arange(0, feature_size), feature_names)
            #ax.set_yticklabels(feature_names, rotation=0)
            if i % num_cols != 0:  # subplot not one of the left hand side
                plt.setp(ax.get_yticklabels(), visible=False)
            if i / num_rows < num_rows-1:  # subplot not in the last row
                plt.setp(ax.get_xticklabels(), visible=False)
        fig.savefig(os.path.join(dir_name, 'weights_{}.png'.format(
            layer_num)), format='png', dpi=200)


def plot_pca(sess, model, feed_dict, colors, dir_name):
    print("Plotting TSNE for activaions...")
    # assign color to the labels
    node_names
    layer_num = 0
    for layer_act in model.activations:
        if layer_num == 0:  # don't plot input distribution
            layer_num += 1
            continue
        else:
            activation = sess.run(layer_act, feed_dict=feed_dict)
            if activation.shape[1] > 1:
                embedding = PCA(n_components=2).fit_transform(activation)
            else:
                continue
        fig = plt.figure(figsize=(14, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1],
                    c=colors.color, alpha=0.7)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA Plot: Hidden Layer {}'.format(layer_num))

        # legend
        pred_nodes = mpatches.Patch(color='blue', label='Predicted Node')
        train_nodes = mpatches.Patch(color='green', label='Training Nodes')
        test_nodes = mpatches.Patch(color='red', label='Test Nodes')
        not_involved = mpatches.Patch(
            color='gray', label='Not Predicted and not labeled')
        plt.legend(handles=[pred_nodes, train_nodes, test_nodes, not_involved])

        # save
        fig.savefig(os.path.join(
            dir_name, 'pca_{}.png'.format(layer_num)), dpi=300)
        print("Plotted TSNE for layer {}".format(layer_num))
        layer_num += 1


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
                        default=None,
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


if __name__ == "__main__":
    args = parse_args()
    if args.data is None or not args.data.endswith('.h5'):
        print("No path to HDF5 data container provided or data is not hdf5. Exit now.")
        sys.exit(-1)

    input_data_path = args.data
    data = load_hdf_data(input_data_path, feature_name='features_mean')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names = data
    print("Read data from: {}".format(input_data_path))
    #data = load_cora()
    #adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = data
    #node_names = np.array([[str(i), str(i)] for i in np.arange(features.shape[0])])
    num_nodes = adj.shape[0]
    num_feat = features.shape[1]
    if num_feat > 1:
        features = gcn.utils.preprocess_features(lil_matrix(features))
    else:
        print("Not row-normalizing features because feature dim is {}".format(num_feat))
        features = gcn.utils.sparse_to_tuple(lil_matrix(features))

    # preprocess adjacency matrix and account for larger support
    poly_support = args.support
    if poly_support > 0:
        support = gcn.utils.chebyshev_polynomials(adj, poly_support)
        num_supports = 1 + poly_support
    else: # support is 0, don't use the network
        support = [sp.eye(adj.shape[0])]
        num_supports = 1

    # create placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32,
                                          shape=features[2]),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)
    }
    hidden_dims = [int(x) for x in args.hidden_dims]

    # create session, train and save afterwards
    #device = 1 if fits_on_gpu(adj, features, hidden_dims, poly_support) else 0
    #print (device)
    #config = tf.ConfigProto(device_count={'GPU': device})
    with tf.Session() as sess:
        model = MYGCN(placeholders=placeholders,
                      input_dim=features[2][1],
                      learning_rate=args.lr,
                      weight_decay=args.decay,
                      num_hidden_layers=len(args.hidden_dims),
                      hidden_dims=hidden_dims,
                      pos_loss_multiplier=args.loss_mul,
                      logging=True)

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
        config = projector.ProjectorConfig()
        train_writer = tf.summary.FileWriter(os.path.join(save_path, 'train'))
        test_writer = tf.summary.FileWriter(os.path.join(save_path, 'test'))

        # helper functions for evaluation at training time
        def evaluate(features, support, labels, mask, placeholders):
            d = gcn.utils.construct_feed_dict(
                features, support, labels, mask, placeholders)
            loss, acc, aupr, auroc = sess.run([model.loss, model.accuracy,
                                               model.aupr_score, model.auroc_score],
                                              feed_dict=d)
            return loss, acc, aupr, auroc

        def predict(features, support, labels, mask, placeholders):
            feed_dict_pred = gcn.utils.construct_feed_dict(
                features, support, labels, mask, placeholders)
            pred = sess.run(model.predict(), feed_dict=feed_dict_pred)
            return pred

        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
        for epoch in range(args.epochs):
            feed_dict = gcn.utils.construct_feed_dict(features, support, y_train,
                                                      train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: args.dropout})
            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy,
                             model.auroc_score, model.aupr_score, merged],
                            feed_dict=feed_dict)
            train_writer.add_summary(outs[5], epoch)
            train_writer.flush()

            # Print results
            if epoch % 5 == 0 or epoch-1 == args.epochs:
                d = gcn.utils.construct_feed_dict(features, support, y_test,
                                                  test_mask, placeholders)
                # loss, acc, aupr, auroc = sess.run([model.loss, model.accuracy,
                #                                   model.aupr_score, model.auroc_score],
                #                                  feed_dict=d)
                summary = sess.run(merged, feed_dict=d)

                test_writer.add_summary(summary, epoch)
                test_writer.flush()
                val_acc = sess.run(model.accuracy, feed_dict=d)
                print("Epoch:", '%04d' % (epoch + 1),
                      "Train Loss=", "{:.5f}".format(outs[1]),
                      "Train Acc=", "{:.5f}".format(outs[2]),
                      "Train AUROC={:.5f}".format(outs[3]),
                      "Train AUPR: {:.5f}".format(outs[4]),
                      "Val Acc=", "{:.5f}".format(val_acc))
            else:
                print("Epoch:", '%04d' % (epoch + 1),
                      "Train Loss=", "{:.5f}".format(outs[1]),
                      "Train Acc=", "{:.5f}".format(outs[2]),
                      "Train AUROC={:.5f}".format(outs[3]),
                      "Train AUPR: {:.5f}".format(outs[4]))
        print("Optimization Finished!")

        # Testing
        test_cost, test_acc, test_aupr, test_auroc = evaluate(
            features, support, y_test, test_mask, placeholders)
        print("Test set results:", "loss=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(
                  test_acc), "aupr=", "{:.5f}".format(test_aupr),
              "auroc=", "{:.5f}".format(test_auroc))

        # add embeddings. This is not optimal here. TODO: Add embeddings in class
        config = projector.ProjectorConfig()
        i = 0
        for output in model.activations[1:]:
            test_dict = gcn.utils.construct_feed_dict(
                features, support, y_test, test_mask, placeholders)
            act = output.eval(feed_dict=test_dict, session=sess)
            embedding_var = tf.Variable(
                act, name='activation_layer_{}'.format(i))
            sess.run(embedding_var.initializer)
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            i += 1
        projector.visualize_embeddings(train_writer, config)

        # predict node classification
        predictions = predict(features, support, y_test,
                              test_mask, placeholders)
        print(predictions.shape)
        # save model
        model_save_path = os.path.join(save_path, 'model.ckpt')
        print("Save model to {}".format(model_save_path))
        path = model.save(model_save_path, sess=sess)
        #saver = tf.train.Saver()
        #path = saver.save(sess, model_save_path)

        # save predictions
        with open(os.path.join(save_path, 'predictions.tsv'), 'w') as f:
            f.write('ID\tName\tProb_pos\n')
            for pred_idx in range(predictions.shape[0]):
                f.write('{}\t{}\t{}\n'.format(node_names[pred_idx, 0],
                                              node_names[pred_idx, 1],
                                              predictions[pred_idx, 0])
                        )
        # construct color DataFrame for PCA plots
        colors = get_color_dataframe(node_names, predictions, y_train, y_test)

        # save hyper Parameters and plot
        write_hyper_params(args, input_data_path, os.path.join(
            save_path, 'hyper_params.txt'))
        plot_weights(sess, model, save_path)
        plot_pca(sess, model, test_dict, colors, save_path)
        plot_roc_pr_curves(
            predictions[test_mask == 1], y_test[test_mask == 1], save_path)
