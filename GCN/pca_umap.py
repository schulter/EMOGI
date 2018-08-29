import numpy as np
from sklearn.decomposition import PCA 
from sklearn import preprocessing
import matplotlib.pyplot as plt
import h5py


def load_hdf_data(path, network_name='network', feature_name='features'):
    with h5py.File(path, 'r') as f:
        network = f[network_name][:]
        features = f[feature_name][:]
        node_names = f['gene_names'][:]
        train_mask = f['mask_train'][:]
        if 'feature_names' in f:
            feature_names = f['feature_names'][:]
        else:
            feature_names = None
        positives = []
        y_train = f['y_train'][:]
        positives.append(y_train.flatten())
        if 'y_test' in f:
            positives.append(f['y_test'][:].flatten())
        if 'y_val' in f:
            positives.append(f['y_val'][:].flatten())
        genes_pos = set()
        for group in positives:
            genes_pos.update(np.nonzero(group)[0])
        #genes_pos = set(node_names[i,1] for i in genes_pos)
    return np.array(list(genes_pos)), feature_names
    #return network, features, y_train, train_mask, node_names, feature_names, genes_pos


def main():
    positives, feature_names = load_hdf_data("../data/pancancer/iref_multiomics_norm_tcgage_methpromonly1000bp.h5", feature_name='features')
    colors = np.ones((12129,))
    colors[positives] = 2

    data = np.load("../data/pancancer/multiomics_features_raw.npy")
    data = np.delete(data, list(range(23,35)),axis=1)
    sums = data.sum(axis=1)
    print(sums.shape)
    is_zero = np.where(np.isclose(sums, np.zeros((len(sums),))))[0]
    print(is_zero.shape)
    data = np.delete(data, is_zero, axis=0)
    data = preprocessing.quantile_transform(data,axis=0)

    colors = np.delete(colors, is_zero, axis=0)
    print(data.shape)

    # plt.violinplot(data)
    # plt.show()
    n_components = 12

    pca = PCA(n_components=n_components)
    pca.fit(data)
    print("Variance explained:", np.cumsum(pca.explained_variance_ratio_))
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # fig, ax = plt.subplots(nrows = n_components, ncols = 1, figsize = (5,20))
    # for pc in range(loadings.shape[1]):
    #     ax[pc].bar(list(range(35)), loadings[:,pc].T)
    # plt.show()

    data_trans = pca.transform(data)
    fig, ax = plt.subplots(nrows = 6, ncols = 6, figsize = (20,20))
    for x in range(35):
        ax.flat[x].scatter(data_trans[:,0], data_trans[:,1], c=data[:,x], s=2)
        ax.flat[x].set_title(feature_names[x])
    ax.flat[35].scatter(data_trans[:,0], data_trans[:,1], c=colors, s=2)
    ax.flat[35].set_title("positive genes")
    plt.tight_layout()
    fig.savefig("PCA.png")

    # plt.violinplot(data_trans)
    # plt.show()

    # import umap
    # trans = umap.UMAP().fit(data)
    # fig, ax = plt.subplots(nrows = 6, ncols = 6, figsize = (20,20))
    # print(dir(trans))
    # for x in range(35):
    #     ax.flat[x].scatter(trans.embedding_[:,0], trans.embedding_[:,1], c=data[:,x], s=2)
    #     ax.flat[x].set_title(feature_names[x])
    # ax.flat[35].scatter(trans.embedding_[:,0], trans.embedding_[:,1], c=colors, s=2)
    # ax.flat[35].set_title("positive genes")
    # plt.tight_layout()
    # fig.savefig("umap.png")

#CUDA_VISIBLE_DEVICES=0 python train_cv.py -e 7000 -cv 5 -hd 50 40 30 20 10 -s 2 -lm 30 -lr 0.001 -do 0.4 -d ../data/pancancer/PCA.h5

if __name__ == "__main__":
    main()