import argparse
import os, sys
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

import gcnIO
import gcnPreprocessing
import utils
from gcn.models import GCN
from emogi import EMOGI
from train_EMOGI import fit_model, predict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress TF info messages


def run_model(session, params, adj, num_cv, features, y, mask, output_dir):
    """
    """
    # compute support matrices
    support, num_supports = utils.get_support_matrices(adj, params['support'])
    # construct placeholders & model
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, name='support_{}'.format(i)) for i in range(num_supports)],
        'features': tf.placeholder(tf.float32, shape=features.shape, name='Features'),
        'labels': tf.placeholder(tf.float32, shape=(None, y.shape[1]), name='Labels'),
        'labels_mask': tf.placeholder(tf.int32, shape=mask.shape, name='LabelsMask'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='Dropout')
    }
    model = EMOGI(placeholders=placeholders,
                  input_dim=features.shape[1],
                  learning_rate=params['learningrate'],
                  weight_decay=params['weight_decay'],
                  num_hidden_layers=len(params['hidden_dims']),
                  hidden_dims=params['hidden_dims'],
                  pos_loss_multiplier=params['loss_mul'],
                  logging=False
    )
    # where the results go
    accs = []
    losses = []
    auprs = []
    num_preds = []

    k_sets = gcnPreprocessing.cross_validation_sets(y=y,
                                                    mask=mask,
                                                    folds=num_cv
    )
    for cv_run in range(num_cv):
        # train model
        y_train, y_val, train_mask, val_mask = k_sets[cv_run]
        model = fit_model(model=model,
                          sess=session,
                          features=features,
                          placeholders=placeholders,
                          support=support,
                          epochs=params['epochs'],
                          dropout_rate=params['dropout'],
                          y_train=y_train,
                          train_mask=train_mask,
                          y_val=y_val,
                          val_mask=val_mask,
                          output_dir=os.path.join(output_dir, 'cv_{}'.format(cv_run))
        )
        # Compute performance on validation set
        performance_ops = model.get_performance_metrics()
        sess.run(tf.local_variables_initializer())
        d = utils.construct_feed_dict(features, support, y_val,
                                      val_mask, placeholders)
        val_performance = sess.run(performance_ops, feed_dict=d)
        predictions = sess.run(model.predict(), feed_dict=d)
        accs.append(val_performance[1])
        losses.append(val_performance[0])
        auprs.append(val_performance[2])
        num_preds.append((predictions > 0.5).sum())
    return accs, losses, num_preds, auprs


def write_hyper_param_dict(params, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)

def load_hyper_param_dict(file_name):
    with open(file_name, 'rb') as f:
        params = pickle.load(f)
    return params

def write_performances(performance, file_name):
    write_hyper_param_dict(performance, file_name)

def load_performances(file_name):
    return load_hyper_param_dict(file_name)

def check_param_already_done(training_dir, params):
    for setting in os.listdir(training_dir):
        if setting.startswith('params'):
            other_params = load_hyper_param_dict(os.path.join(training_dir, setting, 'params.txt'))
            if other_params == params:
                return load_performances(os.path.join(training_dir, setting, 'performance.txt'))
    return None


def parse_args():
    parser = argparse.ArgumentParser(description='Run a grid search over HP combinations using cross-validation.')
    parser.add_argument('-d', '--data', help='Path to the data container',
                        dest='data',
                        type=str,
                        required=True
                        )
    parser.add_argument('-cv', '--cv_runs', help='Number of cross validation runs',
                    dest='cv_runs',
                    default=5,
                    type=int
                    )
    parser.add_argument('-td', '--training_dir', help='Training directory name. If already exists, grid search will continue.',
                    dest='train_dir',
                    default=None,
                    type=str
                    )
    parser.add_argument('-o', '--output', help='Output file. A pickle file with performance for HP combinations.',
                    dest='output_file',
                    required=True,
                    type=str
                    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if not args.data.endswith('.h5'):
        print("Data is not a hdf5 container. Exit now.")
        sys.exit(-1)
    basename_out = os.path.dirname(args.output_file)
    if not os.path.isdir(basename_out) and not os.path.dirname(basename_out) is '':
        print ("Directory {} doesn't exist. Can't write output to {}".format(basename_out,
                                                                             args.output_file)
        )
        sys.exit(-1)

    print ("Loading Data from {}".format(args.data))
    cv_runs = args.cv_runs
    data = gcnIO.load_hdf_data(args.data,
                               feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data
    num_nodes = adj.shape[0]
    num_feat = features.shape[1]

    if num_feat > 1:
        #features = utils.preprocess_features(lil_matrix(features))
        print ("Not row-normalizing...")
        #features = utils.sparse_to_tuple(lil_matrix(features))
    else:
        print ("Not row-normalizing features because feature dim is {}".format(num_feat))
        #features = utils.sparse_to_tuple(lil_matrix(features))

    """
    params = {'support':[1],
              'dropout':[0.5],
              'hidden_dims': [[20, 40], [100, 50], [50, 40, 40, 20],
                              [100, 50, 10], [50, 100], [300, 100]],
              'loss_mul': [10, 30, 45, 60, 90, 150],
              'learningrate':[0.001],
              'epochs':[2000],
              'weight_decay':[5e-3]
              }
    """
    params = {'support':[1],
              'dropout':[.1],
              'hidden_dims':[[50, 40], [40, 20]],
              'loss_mul':[1],
              'learningrate':[.1],
              'epochs':[30],
              'weight_decay':[0.05]
              }

    num_of_settings = len(list(ParameterGrid(params)))
    print ("Grid Search: Trying {} different parameter settings...".format(num_of_settings))
    param_num = 0
    # create output directory or set existing one
    if args.train_dir is None:
        out_dir = gcnIO.create_model_dir()
    elif os.path.isdir(args.train_dir):
        out_dir = args.train_dir
    else:
        print ("Cannot output to {} because it is not a directory.".format(args.train_dir))
        sys.exit(-1)
    # create session, train and save afterwards
    performances = []
    for param_set in list(ParameterGrid(params)):
        performance_already_done = check_param_already_done(out_dir, param_set)
        if not performance_already_done is None:
            p = (performance_already_done['accuracy'], performance_already_done['loss'],
                 performance_already_done['num_predicted'], performance_already_done['aupr'],
                 param_set)
            performances.append(p)
            print ("Combination was already processed earlier.")
        else:
            param_dir = os.path.join(out_dir, 'params_{}'.format(param_num))

            with tf.Session() as sess:
                accs, losses, numpreds, auprs = run_model(sess, param_set, adj, 5,
                                                        features, y_train, train_mask, param_dir)
            performance_dict = {'accuracy':accs, 'loss':losses,
                                'num_predicted':numpreds, 'aupr':auprs}
            performances.append((accs, losses, numpreds, auprs, param_set))
            write_hyper_param_dict(param_set, os.path.join(param_dir, 'params.txt'))
            write_performances(performance_dict, os.path.join(param_dir, 'performance.txt'))
            tf.reset_default_graph()
        param_num += 1
        print ("[{} out of {} combinations]: {}".format(param_num, num_of_settings, param_set))

    # write results from gridsearch to file
    with open(args.output_file, 'wb') as f:
        pickle.dump(performances, f)
