import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
import h5py
import numpy as np
import utils
import tensorflow as tf
import sys
import math
from multiprocessing import Process
from deepexplain.tensorflow import DeepExplain
sys.path.append(os.path.abspath('../GCN'))
from my_gcn import MYGCN


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class LRP:


    def __init__(self, model_dir):
        """ Initalize an LRP object.

        During initialization the HDF5 data used for the model training is being loaded
        and preprocessed (computation of support matrices). All files and folders that are
        later required for the LRP are being checked for existence.

        Parameters
        ----------
        model_dir : str
            Path to a folder containing a trained model and CV subfolders.
        """
        # required files and folders
        self.model_dir = model_dir
        self.out_dir = os.path.join(model_dir, 'lrp_sigmoid')
        self.params_file = os.path.join(model_dir, 'hyper_params.txt')
        self.predictions_file = os.path.join(model_dir, 'ensemble_predictions.tsv')
        self.cv_dirs = [f.path for f in os.scandir(model_dir)
                        if f.is_dir() and f.name.startswith('cv_')]
        # check existence of above files and folders
        self._check_files_and_dirs()
        # load data
        self._load_hyper_params()
        self._load_hdf_data()
        self._preprocess_data()


    def _check_files_and_dirs(self):
        if not os.path.isfile(self.params_file):
            raise RuntimeError("hyper params file not found in model folder.")
        if not os.path.isfile(self.predictions_file):
            raise RuntimeError("ensembl predictions file not found in model folder.")
        if len(self.cv_dirs) == 0:
            raise RuntimeError("No cv subfolders found in model folder.")
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)


    def _load_hyper_params(self):
        with open(self.params_file, "rt") as f:
            lines = f.readlines()
        # hyper params
        self.params = {}
        for line in lines[:-1]:
            key, value = line.split('\t')
            if value.startswith("["):
                self.params[key] = [int(v) for v in re.findall(r'[0-9]+', value)]
            else:
                self.params[key] = utils.str_to_num(value.strip())
        # hdf5 data file
        self.data_file = lines[-1].strip()
        if not os.path.isfile(self.data_file):
            raise RuntimeError("hdf5 file '{}' not found.".format(self.data_file))


    def _load_hdf_data(self):
        with h5py.File(self.data_file, 'r') as f:
            self.network = f['network'][:]
            self.features = f['features'][:]
            self.node_names = [x[1] for x in f['gene_names'][:]]
            self.train_mask = f['mask_train'][:]
            self.feature_names = f['feature_names'][:]
            self.features_raw = f['features_raw'][:]
            self.y_train = f['y_train'][:]
            # collect gene symbols of positive genes
            positives = [self.y_train.flatten(),
                         f['y_test'][:].flatten(),
                         f['y_val'][:].flatten()]
            idx_pos = set()
            for group in positives:
                idx_pos.update(np.nonzero(group)[0])
            self.genes_pos = set(self.node_names[i] for i in idx_pos)


    def _preprocess_data(self):
        # compute support
        if self.params["support"] > 0:
            self.support = utils.chebyshev_polynomials(
                self.network, self.params["support"], sparse=False)
        else:
            self.support = [np.eye(self.network.shape[0])]
        # get predicted probabilities for all genes and CV folds
        self.predicted_probs = []
        with open(self.predictions_file, "rt") as f:
            next(f) # skip header
            for line in f:
                self.predicted_probs.append(line.split('\t'))
        assert len(self.predicted_probs) == self.features.shape[0]
        assert len(self.predicted_probs) == len(set(x[1] for x in self.predicted_probs))


    def _run_deepexplain_single_cv(self, cv_dir, gene_name):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=cv_dir)
        placeholders = {
            'support': [tf.placeholder(tf.float32) for _ in range(len(self.support))],
            'features': tf.placeholder(tf.float32, shape=self.features.shape),
            'labels': tf.placeholder(tf.float32, shape=(None, self.y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)
        }
        with tf.Session() as sess:
            with DeepExplain(session=sess) as de:
                model = MYGCN(placeholders=placeholders,
                              input_dim=self.features.shape[1],
                              learning_rate=self.params['lr'],
                              weight_decay=self.params['decay'],
                              num_hidden_layers=len(self.params['hidden_dims']),
                              hidden_dims=self.params['hidden_dims'],
                              pos_loss_multiplier=self.params['loss_mul'],
                              logging=False, sparse_network=False)
                model.load(ckpt.model_checkpoint_path, sess)
                idx_gene = self.node_names.index(gene_name)
                mask_gene = np.zeros((self.features.shape[0], 1))
                mask_gene[idx_gene] = 1
                attributions =  de.explain(method="elrp",
                                  T=tf.nn.sigmoid(model.outputs) * mask_gene,
                                  X=[placeholders['features'], *placeholders["support"]],
                                  xs=[self.features, *self.support])
        tf.reset_default_graph()
        return attributions


    def _get_direct_neighbors(self, node):
        neighbors = []
        for idx, val in enumerate(self.network[node, :]):
            if math.isclose(val, 1):
                neighbors.append(idx)
        return neighbors


    def _get_top_neighbors(self, idx_gene, attr_mean, attr_std):
        edge_list = []
        nodes_attr = {}
        sources = [idx_gene]
        for support in range(2, len(attr_mean)):
            for source in sources:
                next_neighbors = self._get_direct_neighbors(source)
                for next_node in next_neighbors:
                    if not (next_node, source) in edge_list:
                        edge_list.append((source, next_node))
                        if not next_node in nodes_attr:
                            val_mean = (
                                attr_mean[support][idx_gene, next_node] + attr_mean[support][next_node, idx_gene])/2
                            val_std = (
                                attr_std[support][idx_gene, next_node] + attr_std[support][next_node, idx_gene])/2
                            nodes_attr[next_node] = (val_mean, val_std)
            sources = next_neighbors
        return edge_list, nodes_attr


    def _save_edge_list(self, edge_list, nodes_attr, gene_name):
        out_file = os.path.join(self.out_dir, gene_name+".edgelist")
        with open(out_file, "wt") as out_handle:
            out_handle.write("SOURCE\tTARGET\tLRP_ATTR_TARGET\tLABEL_TARGET\n")
            for edge in edge_list:
                out_handle.write("{}\t{}\t{}\t{}\n".format(
                    *edge, nodes_attr[edge[1]][0], self.node_names[edge[1]]))


    def _save_attribution_plots(self, attributions_mean, attributions_std, nodes_attr, gene_name):
        # get most important neighbors for plot
        nodes_sorted = sorted(nodes_attr.items(), key=lambda x: x[1][0])
        nodes_sorted = [(self.node_names[idx], attr) for idx, attr in nodes_sorted]
        most_important_pos = nodes_sorted[-15:][::-1]
        most_important_neg = nodes_sorted[:15]
        # plot most important neighbors and feature attributions
        for line in self.predicted_probs:
            if line[1] == gene_name:
                # (gene name, true label, mean prediction)
                plot_title = (gene_name, line[2], round(float(line[-2]), 3))
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
        # original feature plot
        idx_gene = self.node_names.index(gene_name)
        x = np.arange(len(self.feature_names))
        ax[0].set_title("{} (label = {}, mean prediction = {})".format(*plot_title))
        ax[0].bar(x, self.features_raw[idx_gene, :], color="#67a9cf")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(self.feature_names)
        ax[0].set_ylabel("feature value")
        # LRP attributions for each feature
        ax[1].bar(x, attributions_mean[0][idx_gene, :], color="#5ab4ac",
                  yerr=attributions_std[0][idx_gene, :])
        ax[1].set_xticklabels(self.feature_names)
        ax[1].set_xticks(x)
        ax[1].set_ylabel("LRP attribution")
        for i, val in enumerate(attributions_mean[0][idx_gene, :]):
            if val < 0:
                ax[1].patches[i].set_facecolor("#d8b365")
        # most important neighbors
        neighbors = most_important_pos + most_important_neg[::-1]
        x = np.arange(len(neighbors))
        ax[2].bar(x, [i[1][0] for i in neighbors], color="#5ab4ac",
                  yerr=[i[1][1] for i in neighbors])
        ax[2].set_xticks(x)
        ax[2].set_xticklabels([i[0] for i in neighbors])
        ax[2].set_ylabel("LRP attribution")
        for i, val in enumerate([i[1][0] for i in neighbors]):
            if val < 0:
                ax[2].patches[i].set_facecolor("#d8b365")
        for i, (gene, val) in enumerate(neighbors):
            if gene in self.genes_pos:
                ax[2].get_xticklabels()[i].set_color("red")
        # finalize
        utils._plot_hide_top_right(ax[0])
        utils._plot_hide_top_right(ax[1])
        utils._plot_hide_top_right(ax[2])
        for axis in ax:
            for tick in axis.get_xticklabels():
                tick.set_rotation(90)
        plt.tight_layout()
        fig.savefig(os.path.join(self.out_dir, gene_name+'.pdf'))
        fig.clf()
        plt.close('all')


    def _compute_lrp_single_gene(self, gene_name, only_attr=False):
        if not gene_name in self.node_names:
            print("'{}' not found in gene list. Skipping.")
            return
        print("Now:", gene_name)
        # compute LRP attributions for every CV fold
        attributions = [[] for _ in range(len(self.support)+1)]
        for cv_dir in self.cv_dirs:
            cv_attributions = self._run_deepexplain_single_cv(cv_dir, gene_name)
            for i in range(len(attributions)):
                attributions[i].append(cv_attributions[i])
        # compute mean and std over CV folds
        attributions = [np.array(x) for x in attributions]
        attributions_std = [np.std(x, axis=0) for x in attributions]
        attributions = [np.mean(x, axis=0) for x in attributions]
        # return attributions if plots are not needed
        if only_attr == True:
            return attributions, attributions_std
        # get and save most import network neighbors
        edge_list, nodes_attr = self._get_top_neighbors(idx_gene=self.node_names.index(gene_name),
                                                        attr_mean=attributions,
                                                        attr_std=attributions_std)
        self._save_edge_list(edge_list, nodes_attr, gene_name)
        # plot feature and neighbor attributions
        self._save_attribution_plots(attributions, attributions_std, nodes_attr, gene_name)


    def plot_lrp(self, gene_names, n_processes=1):
        """ Perform LRP and plot attributions for a list of genes.

        This function performs LRP for a list of genes and saves the feature attributions
        and the attributions of the most import network neighbors as a plot to disk. LRP
        is being performed for all CV folds and means and standard deviations are reported
        in the output. An 'edgelist' file of the network neighborhood including LRP
        attributions that can be loaded into Cytoscape is being created as well.

        Warning: Using multiple processes substantially increases the amount of used RAM.

        Parameters
        ----------
        gene_names : str or [str]
            Gene symbol or a list of gene symbols.
        
        n_processes : int > 0
            Number of genes for which LRP should be computed in parallel.
        """
        if not isinstance(gene_names, list):
            gene_names = [gene_names]
        if n_processes == 1:
            for gene_name in gene_names:
                self._compute_lrp_single_gene(gene_name)
        else:
            # using a Pool results in memory errors, because something is hitting
            # a 32bit integer limit, therefore manual processing is used
            chunks = [gene_names[i:i+n_processes] for i in range(0, len(gene_names), n_processes)]
            for chunk in chunks:
                ps = []
                for gene_name in chunk:
                    ps.append(Process(target=self._compute_lrp_single_gene, args=(gene_name,)))
                    ps[-1].start()
                for p in ps:
                    p.join()


    def compute_lrp(self, gene_name):
        """ Perform LRP for a single gene and return the results.

        LRP is being performed for all CV folds of the gene of interest.

        This function returns 4 objects:
        - mean feature attributions (numpy array of shape (number of features, ))
        - std of feature attributions (numpy array of shape (number of features, ))
        - mean support attributions (list of length number of support containing quadratic numpy arrays)
        - std of support attributions (list of length number of support containing quadratic numpy arrays)

        Parameters
        ----------
        gene_name : str
            A gene symbol.
        """
        attr_mean, attr_std = self._compute_lrp_single_gene(gene_name, only_attr=True)
        features_mean = attr_mean[0][self.node_names.index(gene_name), :]
        features_std = attr_std[0][self.node_names.index(gene_name), :]
        return features_mean, features_std, attr_mean[1:], attr_std[1:]


    def compute_lrp_all_genes(self):
        """ Perform LRP for all genes and save summarized results.

        This function calls compute_lrp() for every gene. Gene order is given by the
        gene_names list from the HDF5 data. The following output files will be created:

        feat_mean_all.npy
            -> mean attributions, matrix of shape (number of genes, number of features),
               each row represents the LRP run of the respective gene
        feat_std_all.npy
            -> std of attributions, rest as above
        support_X_mean_sum.npy
            -> mean support attributions summed over all genes,
               matrix of shape (number of genes, number of genes),
               one output file per level of support (i.e. X == 0, 1, 2, etc.)
        """
        # initialize matrices that will be filled over time
        feat_mean_all = np.zeros(self.features.shape)
        feat_std_all = np.zeros(self.features.shape)
        support_mean_sum = [np.zeros(self.network.shape) for _ in range(len(self.support))]

        # save matrices in numpy format
        def save_to_disk():
            np.save(os.path.join(self.out_dir, "feat_mean_all.npy"), feat_mean_all)
            np.save(os.path.join(self.out_dir, "feat_std_all.npy"), feat_std_all)
            for idx, mat in enumerate(support_mean_sum):
                np.save(os.path.join(self.out_dir, "support_{}_mean_sum.npy".format(idx)), mat)

        # run LRP for every gene and save results into aforementioned matrices
        for idx_g, gene_name in enumerate(self.node_names):
            feat_mean, feat_std, supp_mean, _ = self.compute_lrp(gene_name)
            feat_mean_all[idx_g,:] = feat_mean
            feat_std_all[idx_g,:] = feat_std
            for idx_s in range(len(support_mean_sum)):
                support_mean_sum[idx_s] += supp_mean[idx_s]
            # save progress every 500 genes
            if idx_g > 0 and idx_g % 500 == 0:
                save_to_disk()
                print("{} genes done.".format(idx_g+1))
        # save final results
        save_to_disk()


def main():
    interpreter = LRP(model_dir="/project/lincrnas/roman/diseasegcn/data/GCN/training/2019_02_13_15_36_01/")
    # examples:
    # interpreter.plot_lrp(["STIM1", "TRPC1", "NOS1", "ATP2B4", "ABCC9", "KCNJ11"], n_processes=3)
    # feat_mean, feat_std, support_mean, support_std = interpreter.compute_lrp("TP53")
    # interpreter.compute_lrp_all_genes()

    # TODO
    # parallelize compute_lrp_all_genes somehow?
    # command line interface

if __name__ == "__main__":
    main()