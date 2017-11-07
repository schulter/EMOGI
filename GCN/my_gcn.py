
import tensorflow as tf
from utils import *

import os, h5py
import numpy as np
import pickle
from datetime import datetime
import matplotlib.pyplot as plt


def load_hdf_data(path):
    with h5py.File(path, 'r') as f:
        network = f['network'][:]
        features = f['features'][:]
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
    return network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

class MYGCN:
    def __init__(self, learning_rate=.01, dropout_prob=.5, num_feature_maps=16, num_classes=7):
        self.num_feature_maps = num_feature_maps
        self.dropout_keep_prob = dropout_prob
        self.learning_rate = learning_rate
        self.num_outputs = num_classes

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def glorot(self, shape, name=None):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0/(shape[0]+shape[1]))
        initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def build_model(self, adj, x, N, C):
        """
        Build graph convolutional model.

        Parameters:
        ----------
        adj:            Adjecency matrix, the graph.
                        Shape: (N, N) N = #nodes

        x:              Input matrix, node features.
                        Shape: (N, C) C = #node features
        """
        A = adj
        # calculate first hidden layer
        W_0 = self.glorot((C, self.num_feature_maps), "Weights_Layer_0")
        #W_0 = weight_variable((C, num_feature_maps), "Weights_Layer_0")
        A_x = tf.matmul(A, x)
        H_1_preact = tf.matmul(A_x, W_0)
        H_1 = tf.nn.relu(H_1_preact)
        H_1_dropped = tf.nn.dropout(H_1, self.dropout_keep_prob)

        # calculate second hidden layer (output)
        W_1 = self.glorot((self.num_feature_maps, self.num_outputs), "Weights_Layer_1")
        #W_1 = weight_variable((num_feature_maps, num_outputs), "Weights_layer_1")
        A_H1 = tf.matmul(A, H_1_dropped)
        H_2 = tf.matmul(A_H1, W_1) # shape: (N, num_outputs)
        H_2_dropped = tf.nn.dropout(H_2, self.dropout_keep_prob)
        print ("H_2 shape: {}".format(H_2.shape))

        # no activation for H2 as softmax is added by cross-entropy function
        return H_1, H_2


    def construct_computation_graph(self, adj, x, labels, label_mask, num_nodes, num_feat):
        _, output = self.build_model(adj, x, num_nodes, num_feat)
        loss = self.masked_softmax_cross_entropy(output, labels, label_mask)
        acc = self.masked_accuracy(output, labels, label_mask)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return loss, train_step, output, acc

    def train_model(self, adj, features, train_y, test_y, val_y, train_mask, test_mask, val_mask, num_nodes, num_feat, num_epochs=2):
        keep_prob = tf.placeholder(tf.float32, name="dropout_prob")  # dropout (keep probability)
        G = tf.placeholder(tf.float32, shape=adj.shape)
        F = tf.placeholder(tf.float32, shape=features.shape)
        L = tf.placeholder(tf.float32, shape=train_y.shape)
        M = tf.placeholder(tf.float32, shape=train_mask.shape)
        loss_op, train_op, prediction_op, acc_op = self.construct_computation_graph(G, F, L, M, num_nodes, num_feat)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        print ("Graph successfully constructed! Start training...")

        accuracies_test = []
        accuracies_train = []
        cross_entropy_test = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            with tf.device('/gpu:0'):
                feed_d = {G:adj, F:features, L:train_y, M:train_mask, keep_prob: self.dropout_keep_prob}
                train_loss = sess.run(loss_op, feed_dict=feed_d)
                train_acc = sess.run(acc_op, feed_dict=feed_d)
                print ("Starting training with train acc: {}\tloss: {}".format(train_acc, train_loss))
                for epoch in range(num_epochs):
                    # optimize
                    feed_d = {G:adj, F:features, L:train_y, M:train_mask, keep_prob: self.dropout_keep_prob}
                    sess.run(train_op, feed_dict=feed_d)
                    # validation accuracy
                    if val_y is None:
                        feed_val = {G:adj, F:features, L:test_y, M:test_mask, keep_prob: 1.}
                    else:
                        feed_val = {G:adj, F:features, L:val_y, M:val_mask, keep_prob: 1.}
                    val_acc = sess.run(acc_op, feed_dict=feed_val)
                    # training accuracy
                    feed_train_acc = {G:adj, F:features, L:train_y, M:train_mask, keep_prob: 1.}
                    train_acc = sess.run(acc_op, feed_dict=feed_train_acc)
                    # validation loss
                    val_loss = sess.run(loss_op, feed_dict=feed_val)
                    # training loss
                    train_loss = sess.run(loss_op, feed_dict=feed_train_acc)
                    accuracies_train.append(train_acc)
                    accuracies_test.append(val_acc)
                    cross_entropy_test.append(val_loss)
                    print ("[Epoch {}]\tTraining Acc: {:.2f}\tVal Acc: {:.2f}\tTraining Loss: {:.2f}\tVal Loss: {:.2f}".format(epoch, train_acc, val_acc, train_loss, val_loss))
                feed_d = {G:adj, F:features, L:test_y, M: test_mask}
                test_acc = sess.run(acc_op, feed_d)
                print ("Final Accuracy on Test Set: {}".format(test_acc))
            self.finish_after_training(saver, sess, accuracies_test, accuracies_train, cross_entropy_test)

    def evaluate_model(self, adj, features, test_y, test_mask):
        keep_prob = tf.placeholder(tf.float32, name="dropout_prob")  # dropout (keep probability)
        G = tf.placeholder(tf.float32, shape=adj.shape)
        F = tf.placeholder(tf.float32, shape=features.shape)
        L = tf.placeholder(tf.float32, shape=test_y.shape)
        M = tf.placeholder(tf.float32, shape=test_mask.shape)
        loss_op, train_op, prediction_op, acc_op = self.construct_computation_graph(G, F, L, M, num_nodes, num_feat)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            with tf.device('/gpu:0'):
                feed_d = {G:adj, F:features, L:test_y, M:test_mask, keep_prob: 1.}
                prediction = sess.run(prediction_op, feed_dict=feed_d)
                test_acc = sess.run(acc_op, feed_dict=feed_d)
                test_loss = sess.run(loss_op, feed_dict=feed_d)

        print ("Test Accuracy: {}\tTest Loss: {}".format(test_acc, test_loss))
        return prediction

    def get_hidden_activation(self, adj, x, path):
        G = tf.placeholder(tf.float32, shape=adj.shape)
        F = tf.placeholder(tf.float32, shape=x.shape)
        H_1_op, H_2_op = self.build_model(G, F, adj.shape[0], x.shape[1])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, path)
            print ("Model successfully loaded!")
            H_1, H_2 = sess.run([H_1_op, H_2_op], feed_dict={G:adj, F:x})
        return H_1, H_2


    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


    def preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
        adj_normalized = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        return adj_normalized

    def masked_softmax_cross_entropy(self, preds, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)

        print ("loss shape: {}".format(loss.shape))
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(self, preds, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def finish_after_training(self, saver, sess, accuracies_test, accuracies_train, cross_entropy_test):
        if not os.path.isdir('training'):
            os.mkdir('training')
            print ("Created Training Subdir")
        date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.mkdir('training/' + date_string)
        save_path = 'training/' + date_string + '/'
        print ("Save model to {}".format(save_path + 'model.ckpt'))
        path = saver.save(sess, save_path + 'model.ckpt')

        # save accuracies during training
        with open(save_path + 'accuracies.pkl', 'wb') as f:
            pickle.dump((accuracies_test, accuracies_train, cross_entropy_test), f, pickle.HIGHEST_PROTOCOL)

        print ("Plotting...")
        fig = plt.figure(figsize=(14,8))
        plt.plot(accuracies_test, color='green', label='Accuracy on the test set')
        plt.plot(accuracies_train, color='red', label='Accuracy on the training set')
        plt.legend(loc="lower right")
        fig.savefig(save_path + 'plot.png', dpi=400)


if __name__ == "__main__":
    gcn = MYGCN(num_classes=2)
    #adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_hdf_data('../data/preprocessing/legionella_gcn_input.h5')
    num_nodes = adj.shape[0]
    num_feat = features.shape[1]

    # Some preprocessing
    #features = preprocess_features(features)
    adj = gcn.preprocess_adj(adj)
    adj = adj.todense()
    #features = features.todense()
    adj = np.asarray(adj)
    gcn.train_model(adj, features, y_train, y_test, y_val, train_mask, test_mask, val_mask, num_nodes, num_feat, num_epochs=1)
    #predictions = gcn.evaluate_model(adj, features, y_test, test_mask)
