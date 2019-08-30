import argparse, os, sys
import tensorflow as tf
import utils, gcnIO
from my_gcn import MYGCN

from scipy.sparse import lil_matrix
import scipy.sparse as sp

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector


def predict(sess, model, features, support, labels, mask, placeholders):
    feed_dict_pred = utils.construct_feed_dict(
        features, support, labels, mask, placeholders)
    pred = sess.run(model.predict(), feed_dict=feed_dict_pred)
    return pred


def fit_model(model, sess, features, placeholders,
              support, epochs, dropout_rate, y_train,
              train_mask, y_val, val_mask, output_dir):
    """Fits a constructed model to the data.
    
    Trains a given GCN model for as many epochs as specified using
    the provided session. Metrics (ACC/LOSS/AUROC/AUPR) are monitored
    using the provided training and validation labels and masks.
    Early stopping can be used but seems to have problems with
    restoring the model so far.
    
    Parameters:
    ----------
    model:              An already initialized GCN model
    sess:               A tensorflow session object
    placeholders:       A set of tensorflow variables that was already used
                        to construct the model.
                        Can be fed to the session in order to run TF ops
    support:            The support matrices in sparse format
    epochs:             The number of epochs to train for
    dropout_rate:       The percentage of connections set to 0 during
                        training to increase generalization
    y_train:            Vector of size n where n is the number of nodes.
                        A 1 in the vector stands for a positively labelled
                        node and a 0 for a negatively labelled one
    train_mask:         A vector similar to `y_train` but symbolizes nodes
                        that belong to the training set and 0 denotes all
                        other nodes
    y_val:              Similar vector to `y_train` but for the validation set
    val_mask:           Similar vector to `train_mask` but for the validation
                        set
    
    Returns:
    The trained model.
    """
    model_save_path = os.path.join(output_dir, 'model.ckpt')
    performance_ops = model.get_performance_metrics()
    running_avg_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                            scope="evaluation"
    )
    metric_reset_op = tf.variables_initializer(var_list=running_avg_vars)
    early_stopping_mon = utils.EarlyStoppingMonitor(model=model,
                                                sess=sess,
                                                path=model_save_path,
                                                patience=1
    )

    # initialize writers for TF logs
    merged = tf.summary.merge_all()
    val_summary = tf.summary.merge_all(scope='evaluation')
    config = projector.ProjectorConfig()
    train_writer = tf.summary.FileWriter(os.path.join(output_dir, 'train'))
    test_writer = tf.summary.FileWriter(os.path.join(output_dir, 'test'))

    # initialize TF variables
    sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
    )
    
    # training loop
    save_after_training = True
    for epoch in range(epochs):
        # training step
        feed_dict = utils.construct_feed_dict(features, support, y_train,
                                              train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: dropout_rate})
        for key in ['labels', 'features', 'labels_mask']:
            v = placeholders[key]
            print ("{}\t{}\t{}".format(key, v.get_shape(), feed_dict[v].shape))
        print ("Support: {}".format(len(placeholders['support'])))
        for i in range(len(placeholders['support'])):
            print (tf.contrib.util.constant_value(placeholders['support'][i].shape), feed_dict[placeholders['support'][i]][2])
        print (placeholders.keys)
        _ = sess.run(model.opt_op, feed_dict=feed_dict)
        train_loss, train_acc, train_aupr, train_auroc = sess.run(performance_ops,
                                                                    feed_dict=feed_dict)
        if model.logging:
            s = sess.run(merged, feed_dict=feed_dict)
            train_writer.add_summary(s, epoch)
            train_writer.flush()

        # Print validation accuracy once in a while
        if epoch % 10 == 0 or epoch-1 == epochs:
            if model.logging:
                d = utils.construct_feed_dict(features, support, y_val,
                                              val_mask, placeholders)
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
            """
            if early_stopping_mon.should_stop(val_loss):
                print ("Early Stopping")
                #save_after_training = False
                break
            """
        if model.logging:
            print("Epoch:", '%04d' % (epoch + 1),
                    "Train Loss=", "{:.5f}".format(train_loss),
                    "Train Acc=", "{:.5f}".format(train_acc),
                    "Train AUROC={:.5f}".format(train_auroc),
                    "Train AUPR: {:.5f}".format(train_aupr))
    print("Optimization Finished!")

    # save model or restore best performance from early stopping
    if save_after_training:
        print("Save model to {}".format(model_save_path))
        path = model.save(model_save_path, sess=sess)
    else: # restore early stopping best model
        model = MYGCN(placeholders=placeholders,
                    input_dim=features[2][1],
                    learning_rate=0.1,
                    weight_decay=model.weight_decay,
                    num_hidden_layers=model.num_hidden_layers,
                    hidden_dims=model.hidden_dims,
                    pos_loss_multiplier=model.pos_loss_multiplier,
                    logging=True)
        model.load(model_save_path, sess=sess)
    return model


def train_gcn(data_path, n_support, hidden_dims, learning_rate,
              weight_decay, loss_multiplier, epochs, dropout_rate,
              output_dir, logging=True):
    """Train a GCN from some hyper parameters and a HDF5 container.
    
    Construct and fit a GCN model to data stored in a HDF5 container.
    This function encapsulates all Tensorflow stuff. It preprocesses
    features and the adjacency matrix, sets up a model using TF and
    writes predictions to a file in the output directory.
    
    Parameters:
    ----------
    data_path:          Path to a HDF5 container that contains a network,
                        features and train/test splits as data sets inside.
    n_support:          The order of neighborhoods (sometimes called radius)
                        of each node that is taken into account at each layer
    hidden_dims:        The architecture of the model. A list of integers,
                        specifying the number of convolutional kernels per
                        layer and the number of layers altogether
    learning_rate:      The initial learning rate for the adam optimizer
    weight_decay:       The rate of the weight decay
    loss_multiplier:    Times that a positive node (gene) is counted more often
                        than a negative node. Useful for imbalanced data sets
                        in which one class is much more frequent than the
                        other
    epochs:             Number of epochs to train for
    dropout_rate:       Percentage of connection to be set to 0 during training
                        (not evaluation). Improves generalization and robustness
                        of most neural networks
    output_dir:         The output directory. Tensorboard summaries and predictions
                        will be written there
    logging:            Whether or not learning will be monitored and tensorboard
                        events are saved (default is True)

    Returns:
    Predictions. A probability for each node. This is the final prediction for
    all nodes in the network after training. The predictions are a vector and
    the nodes are in the same order as in the HDF5 container.
    """
    # load data and preprocess it
    input_data_path = data_path
    data = gcnIO.load_hdf_data(input_data_path, feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data
    print("Read data from: {}".format(input_data_path))

    # preprocess features
    num_feat = features.shape[1]
    if num_feat > 1:
        #features = utils.preprocess_features(lil_matrix(features))
        #features = utils.sparse_to_tuple(lil_matrix(features))
        pass
    else:
        print("Not row-normalizing features because feature dim is {}".format(num_feat))
        #features = utils.sparse_to_tuple(lil_matrix(features))

    # get higher support matrices
    support, num_supports = utils.get_support_matrices(adj, n_support)

    # create placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32, shape=support[i][2]) for i in range(num_supports)],
        'features': tf.placeholder(tf.float32, shape=features.shape),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32, shape=train_mask.shape),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32, shape=())
    }
    hidden_dims = [int(x) for x in hidden_dims]

    # start actual tensorflow stuff
    with tf.Session() as sess:
        # initialize model and metrics
        model = MYGCN(placeholders=placeholders,
                      input_dim=features.shape[1],
                      learning_rate=learning_rate,
                      weight_decay=weight_decay,
                      num_hidden_layers=len(hidden_dims),
                      hidden_dims=hidden_dims,
                      pos_loss_multiplier=loss_multiplier,
                      logging=logging
        )
        # fit the model
        model = fit_model(model, sess, features, placeholders,
                          support, epochs, dropout_rate, y_train,
                          train_mask, y_val, val_mask, output_dir)

        # Compute performance on test set
        performance_ops = model.get_performance_metrics()
        sess.run(tf.local_variables_initializer())
        d = utils.construct_feed_dict(features, support, y_test, test_mask, placeholders)
        test_performance = sess.run(performance_ops, feed_dict=d)
        print("Test set results:", "loss=", "{:.5f}".format(test_performance[0]),
              "accuracy=", "{:.5f}".format(
                  test_performance[1]), "aupr=", "{:.5f}".format(test_performance[2]),
              "auroc=", "{:.5f}".format(test_performance[3]))

        # predict all nodes (result from algorithm)
        predictions = predict(sess, model, features, support, y_test,
                              test_mask, placeholders)
        gcnIO.save_predictions(output_dir, node_names, predictions)
    return predictions

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


if __name__ == "__main__":
    args = parse_args()
    if not args.data.endswith('.h5'):
        print("Data is not hdf5 container. Exit now.")
        sys.exit(-1)

    #output_dir = gcnIO.create_model_dir()
    output_dir = '../data/GCN/training/2019_08_30_15_21_15'
    predictions = train_gcn(data_path=args.data,
                            n_support=args.support,
                            hidden_dims=args.hidden_dims,
                            learning_rate=args.lr,
                            weight_decay=args.decay,
                            loss_multiplier=args.loss_mul,
                            epochs=args.epochs,
                            dropout_rate=args.dropout,
                            output_dir=output_dir,
                            logging=True)

    # save hyper Parameters and plot
    gcnIO.write_hyper_params(args, args.data,
                             os.path.join(output_dir, 'hyper_params.txt'))
