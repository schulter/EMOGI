# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:29:41 2018

@author: roman
"""

import argparse, os
import tensorflow as tf
import utils, gcnIO, gcnPreprocessing
from scipy.sparse import lil_matrix
import scipy.sparse as sp
import numpy as np
from train_gcn import *
from my_gcn import MYGCN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser(description='Train GCN model and save to file')
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
    parser.add_argument('-cv', '--cv_runs', help='Number of cross validation runs',
                    dest='cv_runs',
                    default=10,
                    type=int
                    )
    args = parser.parse_args()
    return args



def single_cv_run(session, support, num_supports, features, y_train, y_test, train_mask, test_mask, node_names, feature_names, args, model_dir):
    hidden_dims = [int(x) for x in args.hidden_dims]
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=features[2]),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)
    }
    # construct model (including computation graph)
    model = MYGCN(placeholders=placeholders,
                input_dim=features[2][1],
                learning_rate=args.lr,
                weight_decay=args.decay,
                num_hidden_layers=len(hidden_dims),
                hidden_dims=hidden_dims,
                pos_loss_multiplier=args.loss_mul,
                logging=True
    )
    # fit the model
    model = fit_model(model, session, features, placeholders,
                      support, args.epochs, args.dropout,
                      y_train, train_mask, y_test, test_mask,
                      model_dir)
    # Compute performance on test set
    performance_ops = model.get_performance_metrics()
    session.run(tf.local_variables_initializer())
    d = utils.construct_feed_dict(features, support, y_test,
                                  test_mask, placeholders)
    test_performance = session.run(performance_ops, feed_dict=d)
    print("Validataion/test set results:", "loss=", "{:.5f}".format(test_performance[0]),
        "accuracy=", "{:.5f}".format(
            test_performance[1]), "aupr=", "{:.5f}".format(test_performance[2]),
        "auroc=", "{:.5f}".format(test_performance[3]))

    # predict all nodes (result from algorithm)
    predictions = predict(session, model, features, support, y_test,
                          test_mask, placeholders)
    gcnIO.save_predictions(model_dir, node_names, predictions)
    gcnIO.write_train_test_sets(model_dir, y_train, y_test, train_mask, test_mask)
    return test_performance

def run_all_cvs(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names, args, output_dir):
    # preprocess features
    num_feat = features.shape[1]
    if num_feat > 1:
        #print ("Row-normalization!")
        #features = utils.preprocess_features(lil_matrix(np.abs(features)))
        print ("Not row-normalizing...")
        features = utils.sparse_to_tuple(lil_matrix(features))
    else:
        print("Not row-normalizing features because feature dim is {}".format(num_feat))
        features = utils.sparse_to_tuple(lil_matrix(features))

    # get higher support matrices
    support, num_supports = utils.get_support_matrices(adj, args.support)

    # construct splits for k-fold CV
    y_all = np.logical_or(y_train, y_val)
    mask_all = np.logical_or(train_mask, val_mask)
    k_sets = gcnPreprocessing.cross_validation_sets(y=y_all,
                                                    mask=mask_all,
                                                    folds=args.cv_runs
    )

    performance_measures = []
    for cv_run in range(args.cv_runs):
        model_dir = os.path.join(output_dir, 'cv_{}'.format(cv_run))
        y_tr, y_te, tr_mask, te_mask = k_sets[cv_run]
        with tf.Session() as sess:
            val_performance = single_cv_run(sess, support, num_supports, features, y_tr, y_te, tr_mask, te_mask, node_names, feature_names, args, model_dir)
            performance_measures.append(val_performance)
        tf.reset_default_graph()
    # save hyper Parameters
    gcnIO.write_hyper_params(args, args.data, os.path.join(output_dir, 'hyper_params.txt'))
    return performance_measures


if __name__ == "__main__":
    args = parse_args()

    if not args.data.endswith('.h5'):
        print("Data is not hdf5 container. Exit now.")
        sys.exit(-1)

    output_dir = gcnIO.create_model_dir()

    # load data and preprocess it
    input_data_path = args.data
    data = gcnIO.load_hdf_data(input_data_path, feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data
    print("Read data from: {}".format(input_data_path))
    
    run_all_cvs(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names, args, output_dir)