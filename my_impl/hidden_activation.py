from my_gcn import MYGCN
from utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == "__main__":
    gcn = MYGCN()
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
    adj = gcn.preprocess_adj(adj)
    adj = adj.todense()
    features = features.todense()
    adj = np.asarray(adj)
    H_1, H_2 = gcn.get_hidden_activation(adj, features, 'training/2017_07_31_15_32_56/model.ckpt')
    H_1 = adj.T.dot(H_1) # from spectral to input domain
    H_2 = adj.T.dot(H_2) # from spectral to input domain
    print (H_1.shape)#
    print (H_2.shape)

    H_1_labeled_train = H_1[train_mask]
    H_2_labeled_train = H_2[train_mask]
    H_1_labeled_test = H_1[test_mask]
    H_2_labeled_test = H_2[test_mask]
    tsne = TSNE(n_components=2)
    H_1_lowdim_train = tsne.fit_transform(H_1_labeled_train)
    H_2_lowdim_train = tsne.fit_transform(H_2_labeled_train)
    H_1_lowdim_test = tsne.fit_transform(H_1_labeled_test)
    H_2_lowdim_test = tsne.fit_transform(H_2_labeled_test)
    print (H_1_lowdim_train.shape)
    print (H_2_lowdim_train.shape)

    print (y_train.argmax(axis=0))
    fig = plt.figure(figsize=(40, 40))
    plt.subplot(2, 2, 1)
    plt.scatter(H_1_lowdim_train[:, 0], H_1_lowdim_train[:, 1], c=y_train.argmax(axis=1)[train_mask], cmap=plt.get_cmap('Dark2'), s=50)
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title('TSNE Plot First Hidden Layer')
    plt.subplot(2, 2, 2)
    plt.scatter(H_2_lowdim_train[:, 0], H_2_lowdim_train[:, 1], c=y_train.argmax(axis=1)[train_mask], cmap=plt.get_cmap('Dark2'), s=50)
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title('TSNE Plot Second Hidden Layer')
    fig.savefig('tsne.png')
    plt.subplot(2, 2, 3)
    plt.scatter(H_1_lowdim_test[:, 0], H_1_lowdim_test[:, 1], c=y_test.argmax(axis=1)[test_mask], cmap=plt.get_cmap('Dark2'), s=50)
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title('TSNE Plot First Hidden Layer')
    plt.subplot(2, 2, 4)
    plt.scatter(H_2_lowdim_test[:, 0], H_2_lowdim_test[:, 1], c=y_test.argmax(axis=1)[test_mask], cmap=plt.get_cmap('Dark2'), s=50)
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title('TSNE Plot Second Hidden Layer')
    fig.savefig('tsne.png')
