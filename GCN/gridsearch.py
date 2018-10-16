import argparse
import os
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
from my_gcn import MYGCN
from train_gcn import fit_model, predict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def masked_aupr_score(y_true, y_score, mask):
    y_true_masked = y_true[:, 0][mask > 0.5]
    y_score_masked = y_score[:, 0][mask > 0.5]
    return average_precision_score(y_true=y_true_masked, y_score=y_score_masked)

def evaluate(model, session, features, support, labels, mask, placeholders):
    feed_dict_val = utils.construct_feed_dict(features, support, labels, mask, placeholders)
    loss, acc = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return loss, acc


def run_cv(model, sess, features, num_runs, params, placeholders, support, y, mask):
    """Run one parameter setting with CV and evaluate on validation data.
    """
    # where the results go
    accs = []
    losses = []
    auprs = []
    num_preds = []

    k_sets = gcnPreprocessing.cross_validation_sets(y=y,
                                                    mask=mask,
                                                    folds=num_runs
    )
    for cv_run in range(num_runs):
        # select some training genes randomly
        y_train, y_val, train_mask, val_mask = k_sets[cv_run]


        merged = tf.summary.merge_all()
        sess.run(tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()))
        for epoch in range(params['epochs']):
            feed_dict = utils.construct_feed_dict(features, support, y_train,
                                                      train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: params['dropout']})
            outs = sess.run([model.opt_op],
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
    print ("Val AUPR: {}".format(np.mean(auprs)))
    return accs, losses, num_preds, auprs


def run_model(session, params, adj, num_cv, features, y, mask, output_dir):
    """
    """
    # compute support matrices
    support, num_supports = utils.get_support_matrices(adj, params['support'])
    # construct placeholders & model
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=features[2]),
        'labels': tf.placeholder(tf.float32, shape=(None, y.shape[1])),
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
        # select some training genes randomly
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
    with open(file_name, 'w') as f:
        for k, v in params.items():
            f.write('{}\t{}\n'.format(k, v))
    print("Hyper-Parameters saved to {}".format(file_name))


if __name__ == "__main__":
    print ("Loading Data...")
    cv_runs = 5
    data = gcnIO.load_hdf_data('../data/pancancer/iref_multiomics_norm_methnewpromonly_ncglabels_fpkm.h5',
                               feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data
    num_nodes = adj.shape[0]
    num_feat = features.shape[1]

    if num_feat > 1:
        features = utils.preprocess_features(lil_matrix(features))
    else:
        print ("Not row-normalizing features because feature dim is {}".format(num_feat))
        features = utils.sparse_to_tuple(lil_matrix(features))

    params = {'support':[1, 2],
              'dropout':[0.1, 0.25, .5, 0.75],
              'hidden_dims': [[10, 20, 30, 40, 50], [30, 20, 10, 5, 3],
                              [30, 30, 30, 30, 30], [20, 40, 100, 20, 10],
                              [50, 100], [50, 25, 10], [20, 40, 20]],
              'loss_mul': [20, 30, 40],
              'learningrate':[0.001],
              'epochs':[3000],
              'weight_decay':[5e-4]
              }
    """
    params = {'support':[1, 2],
              'dropout':[.1],
              'hidden_dims':[[50, 40]],
              'loss_mul':[1],
              'learningrate':[.1],
              'epochs':[100],
              'weight_decay':[0.05]
              }
    """

    num_of_settings = len(list(ParameterGrid(params)))
    print ("Grid Search: Trying {} different parameter settings...".format(num_of_settings))
    param_num = 1
    # create session, train and save afterwards
    performances = []
    out_dir = gcnIO.create_model_dir()
    for param_set in list(ParameterGrid(params)):
        param_dir = os.path.join(out_dir, 'params_{}'.format(param_num))
        with tf.Session() as sess:
            accs, losses, numpreds, auprs = run_model(sess, param_set, adj, 5,
                                                      features, y_train, train_mask, param_dir)
        performances.append((accs, losses, numpreds, auprs, param_set))
        write_hyper_param_dict(param_set, os.path.join(param_dir, 'params.txt'))
        print ("[{} out of {} combinations]: {}".format(param_num, num_of_settings, param_set))
        param_num += 1
        tf.reset_default_graph()
    # write results from gridsearch to file
    out_name = '../data/gridsearch/gridsearchcv_results_multiomics_norm.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(performances, f)
