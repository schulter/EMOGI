import argparse, os
import tensorflow as tf
import utils, gcnIO
from my_gcn import MYGCN
#import interpretation

from scipy.sparse import lil_matrix
import scipy.sparse as sp

import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

def interpret_results(model_dir):
    print("Running feature interpretation for {}".format(model_dir))
    genes = ["CEBPB", "CHD1", "CHD3", "CHD4", "TP53", "PADI4", "RBL2",
             "BRCA1", "BRCA2", "NOTCH2", "NOTCH1", "MYOC", "ZNF24", "SIM1",
             "HSP90AA1", "ARNT"]
    interpretation.interpretation(
        model_dir, genes, os.path.join(model_dir, 'lrp'), True)


def predict(sess, model, features, support, labels, mask, placeholders):
    feed_dict_pred = utils.construct_feed_dict(
        features, support, labels, mask, placeholders)
    pred = sess.run(model.predict(), feed_dict=feed_dict_pred)
    return pred


def fit_model(model, sess, features, placeholders,
              support, epochs, dropout_rate, y_train,
              train_mask, y_val, val_mask, output_dir):

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
    else:
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
    # load data and preprocess it
    input_data_path = data_path
    data = gcnIO.load_hdf_data(input_data_path, feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names = data
    print("Read data from: {}".format(input_data_path))

    # preprocess features
    num_feat = features.shape[1]
    if num_feat > 1:
        features = utils.preprocess_features(lil_matrix(features))
    else:
        print("Not row-normalizing features because feature dim is {}".format(num_feat))
        features = utils.sparse_to_tuple(lil_matrix(features))

    # get higher support matrices
    support, num_supports = utils.get_support_matrices(adj, n_support)

    # create placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=features[2]),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)
    }
    hidden_dims = [int(x) for x in hidden_dims]

    # start actual tensorflow stuff
    with tf.Session() as sess:
        # initialize model and metrics
        model = MYGCN(placeholders=placeholders,
                      input_dim=features[2][1],
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

    output_dir = gcnIO.create_model_dir()
    predictions = train_gcn(data_path=args.data,
                            n_support=args.support,
                            hidden_dims=args.hidden_dims,
                            learning_rate=args.lr,
                            weight_decay=args.decay,
                            loss_multiplier=args.loss_mul,
                            epochs=args.epochs,
                            dropout_rate=args.dropout,
                            output_dir=output_dir)

    # save hyper Parameters and plot
    gcnIO.write_hyper_params(args, args.data,
                             os.path.join(output_dir, 'hyper_params.txt'))
    # interpret_results(save_path)
