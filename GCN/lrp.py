import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
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
#sys.path.append(os.path.abspath('../GCN'))
from my_gcn import MYGCN
import argparse


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
            if 'features_raw' in f:
                self.features_raw = f['features_raw'][:]
            else:
                self.features_raw = f['features'][:]
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
            'support': [tf.placeholder(tf.float32, shape=self.support[i].shape) for i in range(len(self.support))],
            'features': tf.placeholder(tf.float32, shape=self.features.shape),
            'labels': tf.placeholder(tf.float32, shape=(None, self.y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32, shape=self.train_mask.shape),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32, shape=())
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
                                  T=tf.nn.sigmoid(model.outputs),
                                  X=[placeholders['features'], *placeholders["support"]],
                                  xs=[self.features, *self.support],
                                  ys=mask_gene)
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



    def _save_attribution_plots(self, attributions_mean, attributions_std, nodes_attr, gene_name, heatmap=True):
        # decide which plots to use based on 3D or 2D tensor
        features_3d = len(self.features_raw.shape) == 3

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
        n_neighbors = 3 # number of top neighbors to plot
        nrows = 3 + n_neighbors + 1
        #fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(12, nrows*2+3))
        fig = plt.figure(figsize=(12, nrows*2+3), frameon=False)
        outer_grid = gridspec.GridSpec(nrows=nrows, ncols=1, figure=fig)
        fig.patch.set_visible(False)
        # original feature plot
        idx_gene = self.node_names.index(gene_name)
        #x = np.arange(len(self.feature_names))
        print (len(self.feature_names))
        s = "{0} (label = {1}, mean prediction = {2}, LRP sum total = {3:.2f}, LRP sum gene = {4:.2f})"
        t = s.format(plot_title[0], plot_title[1], plot_title[2],
                     attributions_mean[0].sum(),
                     attributions_mean[0][idx_gene,:].sum()
        )
        # plot the input values
        if heatmap:
            utils.lrp_heatmap_plot(fig, outer_grid[0],
                                   self.features_raw[idx_gene, :],
                                   self.feature_names,
                                   title=t)
        else:
            utils.lrp_barplot(fig, outer_grid[0], self.features_raw[idx_gene, :].reshape(-1),
                              self.feature_names,
                              y_name='Input Features', title=t)
        # plot LRP attributions for each feature
        if heatmap:
            utils.lrp_heatmap_plot(fig, outer_grid[1], attributions_mean[0][idx_gene, :],
                                   self.feature_names)
        else:
            utils.lrp_barplot(fig, outer_grid[1], attributions_mean[0][idx_gene, :],
                              self.feature_names,
                              attributions_std[0][idx_gene, :],
                              y_name='LRP Contributions')
        
        # most important neighbors according LRP feature matrix (rows with highest sums)
        top_n = 20
        if features_3d:
            sums = attributions_mean[0].sum(axis=(1,2))
        else:
            sums = attributions_mean[0].sum(axis=1)
        idx_top = np.argpartition(sums, -top_n)[-top_n:]
        idx_top = idx_top[np.argsort(sums[idx_top])][::-1]
        x = np.arange(top_n)
        x_labels = [self.node_names[i] for i in idx_top]
        ax = plt.Subplot(fig, outer_grid[2])
        ax.bar(x, sums[idx_top], tick_label=x_labels)
        ax.set_ylabel("LRP sum features")
        for i, gene in enumerate(x_labels):
            if gene == gene_name:
                ax.get_xticklabels()[i].set_color("blue")
            elif gene in self.genes_pos:
                ax.get_xticklabels()[i].set_color("red")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        fig.add_subplot(ax)
    
        # LRP attributions of three most important neighbors according to LRP row sums
        x = np.arange(len(self.feature_names))
        for k, idx in enumerate(idx_top[:n_neighbors]):
            if heatmap:
                utils.lrp_heatmap_plot(fig, outer_grid[3+k],
                                       attributions_mean[0][idx, :].T, self.feature_names,
                                       title=self.node_names[idx])
            else:
                utils.lrp_barplot(fig, outer_grid[3+k], attributions_mean[0][idx, :].reshape(-1),
                                 self.feature_names,
                                 std=attributions_std[0][idx, :].reshape(-1),
                                 y_name='LRP Contributions',
                                 title=self.node_names[idx])
    
        # most important neighbors according to LRP support
        neighbors = most_important_pos + most_important_neg[::-1]
        ax_last = plt.Subplot(fig, outer_grid[-1])
        ax_last.set_title("LRP support matrices")
        x = np.arange(len(neighbors))
        ax_last.bar(x, [i[1][0] for i in neighbors], yerr=[i[1][1] for i in neighbors])
        ax_last.set_xticks(x)
        ax_last.set_xticklabels([i[0] for i in neighbors])
        ax_last.set_ylabel("LRP attribution")
        for i, (gene, val) in enumerate(neighbors):
            if gene in self.genes_pos:
                ax_last.get_xticklabels()[i].set_color("red")
        # finalize
        plt.tight_layout()
        fig.savefig(os.path.join(self.out_dir, gene_name+'.pdf'))
        fig.clf()
        plt.close('all')


    def _compute_lrp_single_gene(self, gene_name, only_attr=False, heatmap=True):
        if not gene_name in self.node_names:
            print("'{}' not found in gene list. Skipping.".format(gene_name))
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
        # get and save most import network neighbors otherwise
        else:
            edge_list, nodes_attr = self._get_top_neighbors(idx_gene=self.node_names.index(gene_name),
                                                            attr_mean=attributions,
                                                            attr_std=attributions_std)
            self._save_edge_list(edge_list, nodes_attr, gene_name)
            # plot feature and neighbor attributions
            self._save_attribution_plots(attributions, attributions_std, nodes_attr, gene_name, heatmap=heatmap)


    def _compute_lrp_all_genes_single_cv(self, cv_dir):
        """Computes LRP for a number of genes for one CV only.

        This function computes the LRP for a (high) number of genes but only for
        a single CV run. This is much faster than computing all 10 runs per
        gene at a time.

        Parameters:
        ----------
        cv_dir : str
            The folder from which to reconstruct the model

        Returns:
        A tuple with feature and neighbor contributions. The feature contributions
        have the same shape as the features and each row contains the LRP for one
        gene in the same order as the node_names in the HDF5 container.
        The neighbor contribution is a list of numpy arrays in the same shape as
        the adjacency matrix of the graph (but is not sparse).


        """
        neighbor_contribution = [np.zeros_like(self.network) for _ in range(self.params["support"] + 1)]
        feature_contribution = np.zeros_like(self.features)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=cv_dir)
        placeholders = {
            'support': [tf.placeholder(tf.float32, shape=self.support[i].shape) for i in range(len(self.support))],
            'features': tf.placeholder(tf.float32, shape=self.features.shape),
            'labels': tf.placeholder(tf.float32, shape=(None, self.y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32, shape=self.train_mask.shape),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32, shape=())
        }
        with tf.Session() as sess:
            with DeepExplain(session=sess) as de:
                # create model and placeholders
                model = MYGCN(placeholders=placeholders,
                              input_dim=self.features.shape[1],
                              learning_rate=self.params['lr'],
                              weight_decay=self.params['decay'],
                              num_hidden_layers=len(self.params['hidden_dims']),
                              hidden_dims=self.params['hidden_dims'],
                              pos_loss_multiplier=self.params['loss_mul'],
                              logging=False, sparse_network=False)
                model.load(ckpt.model_checkpoint_path, sess)

                # get the explainer
                explainer = de.get_explainer(method='elrp',
                                             T=tf.nn.sigmoid(model.outputs),
                                             X=[placeholders['features'], *placeholders["support"]]
                )

                # run the explain function for every gene
                for idx_g, gene_name in enumerate(self.node_names):
                    mask_gene = np.zeros((self.features.shape[0], 1))
                    mask_gene[idx_g] = 1
                    attributions = explainer.run(xs=[self.features, *self.support],
                                                 ys=mask_gene
                    )
                    feature_contribution[idx_g, :] = attributions[0][idx_g, :]
                    # add neighbor contributions for all support matrices
                    for support in range(len(neighbor_contribution)):
                        # divide by total (neighbor) contribution to make
                        # every gene contribute the same total numbers
                        total_neighbor_contrib = np.abs(attributions[support+1].sum().sum())
                        if total_neighbor_contrib > 0.01:
                            #print (total_neighbor_contrib, (attributions[support+1] / total_neighbor_contrib).sum().sum())
                            neighbor_contribution[support] += (attributions[support+1] / total_neighbor_contrib)
                    if idx_g > 0 and idx_g % 500 == 0:
                        print ("Computed LRP for {} genes so far".format(idx_g))
        tf.reset_default_graph()
        return feature_contribution, neighbor_contribution

    def compute_lrp_all_genes_fast(self):
        """Compute the LRP for all genes efficiently.
        The function computes the LRP for all genes by iterating over all
        genes for one CV first and only then iterating over the CV runs.
        This way, the computational graph in TF can be reused and time saved.

        The following output files are created but no values are returned:
        feat_mean_all.npy
            -> mean attributions, matrix of shape (number of genes, number of features),
               each row represents the LRP run of the respective gene
        feat_std_all.npy
            -> standard deviation of attributions, rest as above
        support_X_mean_sum.npy
            -> mean support attributions summed over all genes,
               matrix of shape (number of genes, number of genes),
               one output file per level of support (i.e. X == 0, 1, 2, etc.)
        """
        # the final matrices containing the results in the end
        neighbor_contributions = [[] for _ in range(self.params["support"] + 1)]
        feature_contributions = []
    
        # iterate through CV dirs
        cv_count = 0
        for cv_dir in self.cv_dirs:
            attr_features, attr_neighbors = self._compute_lrp_all_genes_single_cv(cv_dir)
            feature_contributions.append(attr_features)
            for support in range(self.params["support"] + 1):
                neighbor_contributions[support].append(attr_neighbors[support])
            cv_count += 1
            print ("[CV {} finished] Computed LRP for all genes in {}".format(cv_count, cv_dir))
        
        # compute mean and std
        feature_contribution_final = np.mean(feature_contributions, axis=0)
        feature_contribution_std = np.std(feature_contributions, axis=0)
        # since we have s support matrices, we sum over axis 1 to get shape: (s x n x n)
        neighbor_contributions_final = np.mean(neighbor_contributions, axis=1)

        # save to disk
        def save_to_disk():
            np.save(os.path.join(self.out_dir, "feat_mean_all.npy"), feature_contribution_final)
            np.save(os.path.join(self.out_dir, "feat_std_all.npy"), feature_contribution_std)
            for idx, mat in enumerate(neighbor_contributions_final):
                np.save(os.path.join(self.out_dir, "support_{}_mean_sum.npy".format(idx)), mat)
        save_to_disk()

    def plot_lrp(self, gene_names, n_processes=1, heatmap_plots=True):
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
                self._compute_lrp_single_gene(gene_name, heatmap=heatmap_plots)
        else:
            # using a Pool results in memory errors, because something is hitting
            # a 32bit integer limit, therefore manual processing is used
            chunks = [gene_names[i:i+n_processes] for i in range(0, len(gene_names), n_processes)]
            for chunk in chunks:
                ps = []
                for gene_name in chunk:
                    ps.append(Process(target=self._compute_lrp_single_gene, args=(gene_name, heatmap_plots)))
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
            np.save(os.path.join(self.out_dir, "feat_mean_all_new.npy"), feat_mean_all)
            np.save(os.path.join(self.out_dir, "feat_std_all_new.npy"), feat_std_all)
            for idx, mat in enumerate(support_mean_sum):
                np.save(os.path.join(self.out_dir, "support_{}_mean_sum_normed.npy".format(idx)), mat)

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
    parser = argparse.ArgumentParser(
        description='Compute LRP scores for genes and neighbors')
    parser.add_argument('-m', '--modeldir',
                        help='Path to the trained model directory',
                        dest='model_dir',
                        required=True)
    parser.add_argument('-g', '--genes',
                        help='List of gene symbols for which to do interpretation',
                        nargs='+',
                        dest='genes',
                        default=None)

    parser.add_argument('-a', '--all', help='Compute LRP for all genes', dest='all_genes',
                        default=False, type=bool)
    parser.add_argument('-b', '--bars', help='Plot Barplots', dest='bar_plot',
                        default=False, type=bool)
    args = parser.parse_args()

    # decide whether to compute LRP for all genes
    if args.all_genes:
        interpreter = LRP(model_dir=args.model_dir)
        interpreter.compute_lrp_all_genes_fast()

    # get some genes to do interpretation for
    elif args.genes is None:
        genes = ["CEBPB", "CHD1", "CHD3", "CHD4", "TP53", "RBL2", "BRCA1",
                 "BRCA2", "NOTCH2", "NOTCH1", "ZNF24", "SIM1", "HSP90AA1",
                 "ARNT", "KRAS", "SMAD6", "SMAD4",  "STAT1", "MGMT", "NCOR2",
                 "RUNX1", "KAT7", "IDH1", "IDH2", "DROSHA", "WRN", "FOXA1",
                 "RAC1", "BIRC3", "DNM2", "MYC", "BRAF", "EGFR", "FGFR1",
                 "FGFR2", "FGFR3", "TTN", "TWIST1",
                 "APC", "ERBB3", "ERBB2", "AR", "NRAS", "HDAC3"]
        interpreter = LRP(model_dir=args.model_dir)
        interpreter.plot_lrp(genes)
    else:
        genes = args.genes
        interpreter = LRP(model_dir=args.model_dir)
        interpreter.plot_lrp(genes, heatmap_plots=not args.bar_plot)
    # examples:
    # interpreter = LRP(model_dir="/project/lincrnas/roman/diseasegcn/data/GCN/training/2019_03_06_15_45_33/")
    # interpreter.plot_lrp(["TP53", "KRAS", "TTN", "MYC", "TWIST1", "HIST1H3E", "APC"], n_processes=4)
    # feat_mean, feat_std, support_mean, support_std = interpreter.compute_lrp("TP53")
    # interpreter.compute_lrp_all_genes()


if __name__ == "__main__":
    main()
