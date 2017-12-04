import os, h5py
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from gcn.models import GCN, MLP
from gcn.utils import *
import tensorflow as tf
from scipy.sparse import csr_matrix, lil_matrix
import time
from datetime import datetime

flags = tf.app.flags
FLAGS = flags.FLAGS
#flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 150, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 20, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_bool('cheby', True, 'Using Chebyshev convolutions or not.')

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

def load_cora():
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../gcn/gcn/data/ind.cora.{}".format(names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../gcn/gcn/data/ind.cora.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

if __name__ == "__main__":
    print ("Loading Data...")
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names = load_hdf_data('../data/preprocessing/legionella_gcn_input.h5')
    print (node_names.shape)
    print (adj.sum(axis=1).min())
    adj = csr_matrix(adj)
    features = lil_matrix(features)
    #adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_cora()

    print (adj.shape, features.shape)
    print (type(adj), type(features))
    # Some preprocessing
    #adj = preprocess_adj(adj)
    features = preprocess_features(features)
    if FLAGS.cheby:
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
    else:
        support = [preprocess_adj(adj)]
        num_supports = 1
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    model = GCN(placeholders, input_dim=features[2][1], logging=True)

    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time()-t_test)

    def predict(features, support, labels, mask, placeholders):
        feed_dict_pred = construct_feed_dict(features, support, labels, mask, placeholders)
        pred = sess.run(model.predict(), feed_dict=feed_dict_pred)
        return pred

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    cost_val = []
    for epoch in range(FLAGS.epochs):
        t = time.time()
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, duration = evaluate(features, support, y_test, test_mask, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))


        #if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        #    print("Early stopping...")
        #    break
    print("Optimization Finished!")

    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    # do final prediction and save model to directory

    # predict node classification
    predictions = predict(features, support, y_test, test_mask, placeholders)

    if not os.path.isdir('../GCN/training'):
        os.mkdir('../GCN/training')
        print ("Created Training Subdir")

    model.save(sess)
    # save model
    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir('../GCN/training/' + date_string)
    save_path = '../GCN/training/' + date_string + '/'
    print ("Save model to {}".format(save_path + 'model.ckpt'))
    path = saver.save(sess, save_path + 'model.ckpt')
    with open(save_path + 'predictions.tsv', 'w') as f:
        f.write('ID\tName\tProb_pos\tProb_neg\n')
        for pred_idx in range(predictions.shape[0]):
            f.write('{}\t{}\t{}\t{}\n'.format(node_names[pred_idx, 0],
                                              node_names[pred_idx, 1],
                                              predictions[pred_idx,0],
                                              predictions[pred_idx,1])
                    )
