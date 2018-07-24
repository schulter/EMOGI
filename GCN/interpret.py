import tensorflow as tf
import os, sys
import h5py
import gcn.utils
from scipy.sparse import csr_matrix, lil_matrix
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

from sklearn.preprocessing import robust_scale


from deepexplain.tensorflow import DeepExplain

sys.path.append(os.path.abspath('../GCN'))
from my_gcn import MYGCN
import utils

def str_to_num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

def load_hyper_params(model_dir):
    file_name = os.path.join(model_dir, 'hyper_params.txt')
    input_file = None
    with open(file_name, 'r') as f:
        args = {}
        for line in f.readlines():
            if '\t' in line:
                key, value = line.split('\t')
                if value.startswith('['): # list of hidden dimensions
                    f = lambda x: "".join(c for c in x if c not in ['\"', '\'', ' ', '\n', '[', ']']) 
                    l = [int(f(i)) for i in value.split(',')]
                    args[key.strip()] = l
                else:
                    args[key.strip()] = str_to_num(value.strip())
            else:
                input_file = line.strip()
    return args, input_file


model_dir = '../data/GCN/training/simulation_LRP_subsupport/'
args, data_file = load_hyper_params(model_dir)
ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_dir)


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


data = load_hdf_data(data_file, feature_name='features')
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names = data

#import pdb; pdb.set_trace()

num_nodes = adj.shape[0]
num_feat = features.shape[1]
if num_feat > 1:
    features = utils.preprocess_features(lil_matrix(features), sparse=False)
else:
    print("Not row-normalizing features because feature dim is {}".format(num_feat))
    features = gcn.utils.sparse_to_tuple(lil_matrix(features))



poly_support = args["support"]
if poly_support > 0:
    support = utils.chebyshev_polynomials(adj, poly_support, sparse=False)
    support = utils.subtract_lower_support(support)
    num_supports = 1 + poly_support
else:
    support = [sp.eye(adj.shape[0])]
    num_supports = 1


placeholders = {
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=features.shape),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)
}


with tf.Session() as sess:
    with DeepExplain(session=sess) as de:
        model = MYGCN(placeholders=placeholders,
                      input_dim=features.shape[1],
                      learning_rate=args['lr'],
                      weight_decay=args['decay'],
                      num_hidden_layers=len(args['hidden_dims']),
                      hidden_dims=args['hidden_dims'],
                      pos_loss_multiplier=args['loss_mul'],
                      logging=True)
        model.load(ckpt.model_checkpoint_path, sess)

        weight_matrices = []
        for var in model.vars: # chebychev coefficients
            weight_matrices.append(model.vars[var].eval(session=sess))
        feed_dict = gcn.utils.construct_feed_dict(features=features,
                                            support=support,
                                            labels=y_train,
                                            labels_mask=train_mask,
                                            placeholders=placeholders
                                        )

        activations = []
        for layer_act in model.activations:
            activation = sess.run(layer_act, feed_dict=feed_dict)
            activations.append(activation)
        
        method = "elrp"
        mask_gene = np.zeros((features.shape[0],1))   # target a specific gene only
        mask_gene[1020] = 1                              # e.g. target gene 1020

        attributions = de.explain(
                    method,                                               #method
                    model.outputs * mask_gene,                            #target tensor
                    [placeholders['features'], *placeholders["support"]], #input tensor
                    [features, *support])                                 #input data
        
        print(len(attributions))
        for x in range(len(attributions)):
            print(x, attributions[x].shape)
        
        #import pdb; pdb.set_trace()
        def plot_support(data, file_name):
            color_map = "Purples"
            interpolation = "none"
            ##fig, ax = plt.subplots(figsize=(30,30))
            plt.matshow(data, aspect="auto", cmap=color_map, interpolation=interpolation)
            plt.savefig(file_name, bbox_inches="tight")
            #ax = sns.heatmap(data)
            #ax.get_figure().savefig(file_name, bbox_inches="tight")

        def plot_features(data, file_name):
            color_map = "Purples"
            interpolation = "none"
            #fig, ax = plt.subplots(figsize=(5,25))
            plt.matshow(data, aspect="auto", cmap=color_map, interpolation=interpolation)
            plt.savefig(file_name, bbox_inches="tight")


        plot_features(robust_scale(attributions[0], axis=1), "interpret_features_{}.pdf".format(method))
        plot_features(robust_scale(features, axis=1), "input_features.pdf")
        plot_support(attributions[1], "interpret_support0_{}.pdf".format(method))
        plot_support(attributions[2], "interpret_support1_{}.pdf".format(method))
        plot_support(attributions[3], "interpret_support2_{}.pdf".format(method))
        plot_support(support[0], "input_support0.pdf")
        plot_support(support[1], "input_support1.pdf")
        plot_support(support[2], "input_support2.pdf")


# import math

# supp = attributions[3]
# a = list(enumerate(supp[1020,:]))
# a = sorted(a, key=lambda x: x[1])
# for pair in a:
#     if not math.isclose(pair[1], 0):
#         print(pair)



# gene = attributions[0][535,:]
# for i, feat in enumerate(gene):
#     print(i, feat)



# feat = attributions[0][:,4]
# feat = list(enumerate(feat))
# a = sorted(feat, key=lambda x: x[1])
# print(a)



#import pdb; pdb.set_trace()
print("done")
#exec(open("interpret.py").read(), globals())



