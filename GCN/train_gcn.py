import argparse
import os, h5py
from datetime import datetime
import tensorflow as tf
import gcn.utils
from my_gcn import MYGCN
from scipy.sparse import lil_matrix
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from tensorflow.contrib.tensorboard.plugins import projector
bestSplit = lambda x: (round(math.sqrt(x)), math.ceil(x / round(math.sqrt(x))))


def load_hdf_data(path, feature_name='features'):
    with h5py.File(path, 'r') as f:
        network = f['network'][:]
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
            feature_names = ['filter_{}'.format(i) for i in range(feature_size)]

        # plot figure itself
        fig = plt.figure(figsize=(30, 20))
        plt.gcf().subplots_adjust(left=0.2)
        num_rows, num_cols = bestSplit(embed_size)
        for i in range(embed_size):
            if i > 1:
                ax = plt.subplot(num_rows, num_cols, i+1, sharey=ax, sharex=ax)
            else:
                ax = plt.subplot(num_rows, num_cols, i+1)
            plt.barh(np.arange(0, feature_size), weight_mat[:,i])
            plt.yticks(np.arange(0, feature_size), feature_names)
            #ax.set_yticklabels(feature_names, rotation=0)
            if i % num_cols != 0: # subplot not one of the left hand side
                plt.setp(ax.get_yticklabels(), visible=False)
            if i / num_rows < num_rows-1: # subplot not in the last row
                plt.setp(ax.get_xticklabels(), visible=False)
        fig.savefig(os.path.join(dir_name, 'weights_{}.png'.format(layer_num)), format='png', dpi=200)


def write_hyper_params(args, file_name):
    with open(file_name, 'w') as f:
        for arg in vars(args):
            f.write('{}\t{}\n'.format(arg, getattr(args, arg)))
    print ("Hyper-Parameters saved to {}".format(file_name))

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
    parser.add_argument('-h1', '--hidden_units_1', help='Number of feature maps for layer 1',
                        dest='hidden1',
                        default=20,
                        type=int
                        )
    parser.add_argument('-h2', '--hidden_units_2', help='Number of feature maps for layer 2',
                        dest='hidden2',
                        default=40,
                        type=int
                        )
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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print ("Loading Data...")
    args = parse_args()
    data = load_hdf_data('../data/simulation/simulated_input_legionella.h5')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names = data
    num_nodes = adj.shape[0]
    num_feat = features.shape[1]
    print (adj.shape, features.shape)

    features = gcn.utils.preprocess_features(lil_matrix(features))

    # create placeholders and other stuff for TF
    poly_support = args.support
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

    # create session, train and save afterwards
    with tf.Session() as sess:
        model = MYGCN(placeholders=placeholders,
                      input_dim=features[2][1],
                      learning_rate=args.lr,
                      weight_decay=args.decay,
                      num_hidden1=args.hidden1,
                      num_hidden2=args.hidden2,
                      pos_loss_multiplier=args.loss_mul,
                      logging=True)

        # create model directory for saving
        root_dir = '../data/GCN/training'
        if not os.path.isdir(root_dir): # in case training root doesn't exist
            os.mkdir(root_dir)
            print ("Created Training Subdir")
        date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.mkdir(os.path.join(root_dir, date_string))
        save_path = os.path.join(root_dir, date_string)

        # initialize writers for TF logs
        merged = tf.summary.merge_all()
        config = projector.ProjectorConfig()
        writer = tf.summary.FileWriter(save_path, sess.graph)

        def evaluate(features, support, labels, mask, placeholders):
            feed_dict_val = gcn.utils.construct_feed_dict(features, support, labels, mask, placeholders)
            loss, acc, aupr, auroc = sess.run([model.loss, model.accuracy, model.aupr_score, model.auroc_score],
                                       feed_dict=feed_dict_val)
            return loss, acc, aupr, auroc

        def predict(features, support, labels, mask, placeholders):
            feed_dict_pred = gcn.utils.construct_feed_dict(features, support, labels, mask, placeholders)
            pred = sess.run(model.predict(), feed_dict=feed_dict_pred)
            return pred

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        cost_val = []
        for epoch in range(args.epochs):
            t = time.time()
            feed_dict = gcn.utils.construct_feed_dict(features, support, y_train, train_mask, placeholders)
            feed_dict.update({placeholders['dropout']: args.dropout})
            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy, merged],
                            feed_dict=feed_dict)
            writer.add_summary(outs[3], epoch)

            # Validation
            cost, acc, aupr, auroc = evaluate(features, support, y_test, test_mask, placeholders)
            cost_val.append(cost)

            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "test_AUPR=", "{:.5f}".format(aupr),
                  "test_AUROC=", "{:.5f}".format(auroc))
        print("Optimization Finished!")

        # Testing
        test_cost, test_acc, test_aupr, test_auroc = evaluate(features, support, y_test, test_mask, placeholders)
        print("Test set results:", "loss=", "{:.5f}".format(test_cost),
        "accuracy=", "{:.5f}".format(test_acc), "aupr=", "{:.5f}".format(test_aupr),
        "auroc=", "{:.5f}".format(test_auroc))

        # add embeddings. This is not optimal here. TODO: Add embeddings in class
        config = projector.ProjectorConfig()
        i = 0
        for output in model.activations[1:]:
            test_dict = gcn.utils.construct_feed_dict(features, support, y_test, test_mask, placeholders)
            print (type(output))
            act = output.eval(feed_dict=test_dict, session=sess)
            print (i, act.shape, type(act))
            embedding_var = tf.Variable(act, name='embedding_{}'.format(i))
            sess.run(embedding_var.initializer)
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            i += 1
        projector.visualize_embeddings(writer, config)
        
        # predict node classification
        predictions = predict(features, support, y_test, test_mask, placeholders)

        # save model
        model_save_path = os.path.join(save_path, 'model.ckpt')
        print ("Save model to {}".format(model_save_path))
        #path = model.save(model_save_path, sess=sess)
        saver = tf.train.Saver()
        path = saver.save(sess, model_save_path)
        

        # save predictions
        with open(os.path.join(save_path, 'predictions.tsv'), 'w') as f:
            f.write('ID\tName\tProb_pos\tProb_neg\n')
            for pred_idx in range(predictions.shape[0]):
                f.write('{}\t{}\t{}\t{}\n'.format(node_names[pred_idx, 0],
                                                  node_names[pred_idx, 1],
                                                  predictions[pred_idx,0],
                                                  predictions[pred_idx,1])
                        )

        # save hyper Parameters
        write_hyper_params(args, os.path.join(save_path, 'hyper_params.txt'))
        plot_weights(sess, model, save_path)
