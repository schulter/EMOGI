# EMOGI: Explainable Multi-Omics Graph Integration
This project predicts cancer genes based on multi-omics feature vectors and protein-protein interactions. Each gene is a data point/node and semi-supervised graph convolutional networks are used for classifying cancer genes.

## Installation
Requires the following python packages:
* Tensorflow
* h5py
* Networkx
* deepExplain
* mygene
* Matplotlib
* Seaborn
* Sklearn
* Scipy
* ...

## Computing Contributions for Genes of Interest
A trained EMOGI model can be interrogated in a gene-wise fashion to find out why 
To compute the feature and interaction partner contributions for a gene of interest, use:
```
python lrp.py -m <path-to-model-directory> -g <hugo-symbol1> <hugo-symbol2> -b True/False
```
The genes have to be provided as hugo gene symbols, eg. `EGFR` or `BRCA1`. The `-b` option controls if the resulting plots are heatmaps (more compact, as shown in the paper) or more informative barplots whose error bars indicate standard deviation across cross-validation runs.
Finally, to compute the contributions for all genes (~13,000 to 15,000 depending on the PPI network used), you can specify the `-a` option. This will take a long time, however, and uses all available cores.

*Note: The LRP script is often better not executed on GPU because it doesn't benefit from it and uses a lot of space.*

## Training EMOGI with Own Data
To train EMOGI with your own data, you have to provide a [HDF5](https://www.h5py.org/) container containing the graph, features and labels. There are scripts in `pancancer/preprocessing` that help with the contruction of the container. In general, a valid container for EMOGI has to contain a graph (called `network`, a numpy matrix of shape N x N), a feature matrix for the nodes (called `features`, of shape N x p), the gene names and IDs (called `gene_names`, as numpy array, dtype `object`), the training set (called `ỳ_train`, as boolean indicator array of shape N x 1), the test set of the same shape (called `y_test`), training and test masks (called `train_mask` and `test_mask`, again, indicator arrays of shape N x 1) and the names of the features (called `feature_names`, an array of length p). Optionally, the raw features can also be added to aid biological interpretation when analyzing the LRP plots (called `features_raw`, the first row plots the input features and it might be better to plot unnormalized values).

Once you obtained a valid HDF5 container, you can simply train EMOGI with
```
python train_cv.py -d <path-to-hdf5-container> -hd <n_filters_layer1> <n_filters_layer2>
```
where the `-hd` argument specifies the number of graph convolutional layers (the number of arguments) and the number of filters per layer. Training with `-hd 100 50` for instance implies that EMOGI is trained with 2 graph convolutional layers with 100 and 50 filters.

## Preparing Own Data for Use with EMOGI
See the [readme file in pancancer](pancancer/README.md) for explanations on how you can process your own data and prepare it for EMOGI training.
