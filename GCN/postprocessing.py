# classics
import numpy as np
import pandas as pd
import h5py
import argparse

# my tool and sparse stuff for feature extraction
import utils, gcnIO
import sys, os
from scipy import interp

sys.path.append(os.path.abspath('../pagerank'))
import pagerank

# sklearn imports
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from functools import reduce

# plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib_venn
plt.rc('font', family='Times New Roman')

# set options
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def compute_ensemble_predictions(model_dir):
    """Computes the mean prediction from cross validation runs.

    This function summarizes the predictions of a GCN between different cross
    validation runs and writes the result into a csv file.

    Parameters:
    ----------
    model_dir:                  The output directory of the GCN training
    """
    args, data_file = gcnIO.load_hyper_params(model_dir)
    if os.path.isdir(data_file): # FIXME: This is hacky and not guaranteed to work at all!
        network_name = None
        for f in os.listdir(data_file):
            if network_name is None:
                network_name = f.split('_')[0].upper()
            else:
                assert (f.split('_')[0].upper() == network_name)
        fname = '{}_{}.h5'.format(network_name, model_dir.strip('/').split('/')[-1])
        data_file = os.path.join(data_file, fname)

    data = gcnIO.load_hdf_data(data_file)
    network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data
    pred_all = []
    sets_all = []
    no_cv = 0
    for cv_dir in os.listdir(model_dir):
        if cv_dir.startswith('cv_'):
            predictions = pd.DataFrame.from_csv(os.path.join(model_dir, cv_dir, 'predictions.tsv'), sep='\t', header=0)
            pred_all.append(predictions)
            y_train_loc, y_test_loc, train_mask_loc, test_mask_loc = gcnIO.read_train_test_sets(os.path.join(model_dir, cv_dir))
            sets_all.append((y_train_loc, y_test_loc, train_mask_loc, test_mask_loc))
            no_cv += 1
    print ("Read predictions from {} CV runs".format(no_cv))

    # get node names
    nodes = pd.DataFrame(node_names, columns=['ID', 'Name'])
    nodes['label'] = np.logical_or(np.logical_or(y_train, y_test), y_val)
    # construct ensemble data frame
    ensemble_predictions = reduce(lambda left,right: pd.merge(left,right,on='Name'), pred_all)
    # get the names corrected
    ensemble_predictions.columns = ['Name'] + ['Prob_pos_{}'.format(i) for i in range(1, no_cv+1)]
    # restore the IDs which were lost during merging (together with the correct order)
    ensemble_predictions = nodes.merge(ensemble_predictions, left_on='Name', right_on='Name')
    ensemble_predictions.set_index('ID', inplace=True)
    # add columns for mean statistics (number predictions, mean prediction, std)
    number_cols = [i for i in ensemble_predictions.columns if i.startswith('Prob_pos_')]
    ensemble_predictions['Num_Pos'] = (ensemble_predictions[number_cols] > 0.5).sum(axis=1)
    ensemble_predictions['Mean_Pred'] = ensemble_predictions[number_cols].mean(axis=1)
    ensemble_predictions['Std_Pred'] = ensemble_predictions[number_cols].std(axis=1)
    # write to file
    predictions = ensemble_predictions.sort_values(by='Mean_Pred', ascending=False)
    predictions.to_csv(os.path.join(model_dir, 'ensemble_predictions.tsv'), sep='\t')
    return pred_all, sets_all

def compute_average_ROC_curve(model_dir, pred_all, sets_all):
    """Computes the average ROC curve across the different cross validation runs.

    This function uses the results from compute_ensemble_predictions and plots
    a ROC curve from that information. The curve depicts also confidence
    intervalls and the average curve as well as AUROC values.

    Parameters:
    ----------
    model_dir:                  The output directory of the GCN training
    pred_all:                   A list with the predictions for all folds
                                (First return value of compute_ensemble_predictions)
    sets_all:                   The different test sets for all folds as list
                                (Second return value of compute_ensemble_predictions)
    """
    # construct test set statistics
    fig = plt.figure(figsize=(20, 12))

    k = 1
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    for pred in pred_all:
        y_t = sets_all[k-1][1]
        m_t = sets_all[k-1][3]
        fpr, tpr, _ = roc_curve(y_score=pred[m_t].Prob_pos, y_true=y_t[m_t])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        auroc = roc_auc_score(y_score=pred[m_t].Prob_pos, y_true=y_t[m_t])
        aucs.append(auroc)
        plt.plot(fpr, tpr, lw=4, alpha=0.3, label='Fold %d (AUROC = %0.2f)' % (k, auroc))
        k += 1

    # plot random line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=3, color='darkred',
            label='Random', alpha=.8)

    # plot mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='navy',
            label=r'Mean ROC curve(AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=6, alpha=.8)

    # plot std dev
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", prop={'size': 20})
    plt.tick_params(axis='both', labelsize=20)
    fig.savefig(os.path.join(model_dir, 'mean_ROC_curve.svg'))
    fig.savefig(os.path.join(model_dir, 'mean_ROC_curve.png'), dpi=300)


def compute_average_PR_curve(model_dir, pred_all, sets_all):
    """Computes the average PR curve across the different cross validation runs.

    This function uses the results from compute_ensemble_predictions and plots
    a ROC curve from that information. The curve depicts also confidence
    intervalls and the average curve as well as AUPR values.

    Parameters:
    ----------
    model_dir:                  The output directory of the GCN training
    pred_all:                   A list with the predictions for all folds
                                (First return value of compute_ensemble_predictions)
    sets_all:                   The different test sets for all folds as list
                                (Second return value of compute_ensemble_predictions)
    """
    fig = plt.figure(figsize=(20, 12))

    k = 1
    y_true = []
    y_pred = []
    pr_values = []
    rec_values = []
    sample_thresholds = np.linspace(0, 1, 100)
    no_pos = []
    no_total = []
    for pred in pred_all:
        y_t = sets_all[k-1][1] # test labels
        m_t = sets_all[k-1][3] # test mask
        pr, rec, thr = precision_recall_curve(probas_pred=pred[m_t].Prob_pos, y_true=y_t[m_t])
        no_pos.append(y_t.sum())
        no_total.append(m_t.sum())
        pr_values.append(interp(sample_thresholds, thr, pr[:-1]))
        #pr_values[-1][-1] = 1.0
        rec_values.append(interp(sample_thresholds, thr, rec[:-1]))
        aupr = average_precision_score(y_score=pred[m_t].Prob_pos, y_true=y_t[m_t])
        plt.plot(rec, pr, lw=4, alpha=0.3, label='Fold %d (AUPR = %0.2f)' % (k, aupr))
        y_true.append(y_t[m_t])
        y_pred.append(pred[m_t].Prob_pos)
        k += 1

    # plot random line
    rand_perf = np.mean(no_pos) / np.mean(no_total)
    plt.plot([0, 1], [rand_perf, rand_perf], linestyle='--', lw=3, color='darkred',
            label='Random', alpha=.8)

    # plot mean curve (PR curve over all folds)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    mean_precision, mean_recall, mean_thresholds = precision_recall_curve(y_true, y_pred)
    label = 'Mean PR (AUPR=%.2f)' % (auc(mean_recall, mean_precision))
    plt.plot(mean_recall, mean_precision, label=label, lw=6, color='navy')

    # plot std dev
    std_pr = np.std(pr_values, axis=0)
    mean_pr = np.mean(pr_values, axis=0)
    mean_rec = np.mean(rec_values, axis=0)
    pr_upper = np.minimum(mean_pr + std_pr, 1)
    pr_lower = np.maximum(mean_pr - std_pr, 0)
    pr_upper = np.append(pr_upper, 1.)
    pr_lower = np.append(pr_lower, 1.)
    mean_rec = np.append(mean_rec, 0.)

    plt.fill_between(mean_rec, pr_lower, pr_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.tick_params(axis='both', labelsize=20)
    plt.xlabel('Recall', fontsize=25)
    plt.ylabel('Precision', fontsize=25)
    plt.legend(prop={'size':20})
    fig.savefig(os.path.join(model_dir, 'mean_PR_curve.svg'))
    fig.savefig(os.path.join(model_dir, 'mean_PR_curve.png'), dpi=300)



def compute_ROC_PR_competitors(model_dir, network_name, verbose=False):
    """Computes ROC and PR curves and compares to base line methods.

    This function uses the mean prediction scores and the test set which was
    not used during training to evaluate the performance of the GCN.
    It computes performance metrics for different base line classifiers
    such as pagerank, random forest, logistic regression and a heat
    diffusion method (inspired from HotNet2).
    It returns the optimal cutoff values from both curves, that is the
    cutoffs at which the GCN achieves the best balance of precision
    and recall (TPR and FPR for the ROC curve).

    Parameters:
    ----------
    model_dir:                  The output directory of the GCN training

    Returns:
    Two scalar values representing the optimal cutoff according to the ROC
    curve and according to the PR curve, respectively.
    """
    # first, get the data from the container
    args, data_file = gcnIO.load_hyper_params(model_dir)
    if os.path.isdir(data_file): # FIXME: This is hacky and not guaranteed to work at all!
        network_name = None
        for f in os.listdir(data_file):
            if network_name is None:
                network_name = f.split('_')[0].upper()
            else:
                assert (f.split('_')[0].upper() == network_name)
        fname = '{}_{}.h5'.format(network_name, model_dir.strip('/').split('/')[-1])
        data_file = os.path.join(data_file, fname)
    data = gcnIO.load_hdf_data(data_file)
    network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data
    # read predictions, too
    predictions = load_predictions(model_dir)

    # prepare the features for easy usage of scikit learn API
    X_train = features[train_mask.astype(np.bool)]
    y_train_svm = y_train[train_mask.astype(np.bool)]
    X_test = features[test_mask.astype(np.bool)]
    y_test_svm = y_test[test_mask.astype(np.bool)]

    # train random forest on the features only and predict for test set and all genes
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train_svm.reshape(-1))
    pred_rf = rf.predict_proba(X_test)
    if verbose: print ("Number of predicted genes in Test set (RF): {}".format(pred_rf.argmax(axis=1).sum()))
    pred_rf_all = rf.predict_proba(features)
    if verbose: print ("RF predicts {} genes in total".format(np.argmax(pred_rf_all, axis=1).sum()))

    # train logistic regression on the features only and predict for test set and all genes
    logreg = LogisticRegression(class_weight='balanced')
    logreg.fit(X_train, y_train_svm.reshape(-1))
    pred_lr = logreg.predict_proba(X_test)
    if verbose: print ("Number of predicted genes in Test set (LogReg): {}".format(pred_lr.argmax(axis=1).sum()))
    pred_lr_all = logreg.predict_proba(features)
    if verbose: print ("LogReg predicts {} genes in total".format(np.argmax(pred_lr_all, axis=1).sum()))

    # train logistic regression on deepWalk embeddings
    fname_dw = '../data/pancancer/deepWalk_results/{}_embedding_CPDBparams.embedding'.format(network_name.upper())
    deepwalk_embeddings = pd.read_csv(fname_dw, header=None, skiprows=1, sep=' ')
    deepwalk_embeddings.columns = ['Node_Id'] + deepwalk_embeddings.columns[1:].tolist()
    deepwalk_embeddings.set_index('Node_Id', inplace=True)
    n_df = pd.DataFrame(node_names, columns=['ID', 'Name'])
    embedding_with_names = deepwalk_embeddings.join(n_df)
    X_dw = embedding_with_names.set_index('Name').reindex(n_df.Name).drop('ID', axis=1)
    X_train_dw = X_dw[train_mask.astype(np.bool)]
    X_test_dw = X_dw[test_mask.astype(np.bool)]
    clf_dw = SVC(kernel='rbf', class_weight='balanced', probability=True)
    clf_dw.fit(X_train_dw, y_train_svm.reshape(-1))
    pred_deepwalk = clf_dw.predict_proba(X_test_dw)

    # load results from graph attention networks (GAT)
    if network_name.upper() == 'CPDB':
        gat_results = np.load('../data/pancancer/gat_results/results_GAT_attention8_CPDB.npy')
    elif network_name.upper() == 'IREF':
        gat_results = np.load('../data/pancancer/gat_results/results_GAT_IREF.npy')
    elif network_name.upper() == 'MULTINET':
        gat_results = np.load('../data/pancancer/gat_results/results_GAT_MULTINET.npy')
    gat_results = gat_results.reshape(gat_results.shape[1], gat_results.shape[2])
    gat_results_test = gat_results[test_mask == 1, :]
    
    # train pagerank on the network
    scores, names = pagerank.pagerank(network, node_names)
    pr_df = pd.DataFrame(scores, columns=['Number', 'Score']) # get the results in same order as our data
    names = pd.DataFrame(names, columns=['ID', 'Name'])
    pr_pred_all = pr_df.join(names, on='Number', how='inner')
    pr_pred_all.drop_duplicates(subset='Name', inplace=True)
    node_names_df = pd.DataFrame(node_names, columns=['ID', 'Name']).set_index('ID')
    pr_pred_all = pr_pred_all.set_index('Name').reindex(node_names_df.Name)
    pr_pred_test = pr_pred_all[pr_pred_all.index.isin(node_names[test_mask == 1, 1])]
    pr_pred_test.drop_duplicates(inplace=True)

    # do a random walk with restart and use HotNet2 heat as p_0
    # read heat json from file
    heat_df = pd.read_json('../../hotnet2/heat_syn_cnasnv.json').drop('parameters', axis=1)
    heat_df.dropna(axis=0, inplace=True)

    # join with node names to get correct order and only genes present in network
    nn = pd.DataFrame(node_names, columns=['ID', 'Name'])
    heat_df = nn.merge(heat_df, left_on='Name', right_index=True, how='left')
    heat_df.fillna(0, inplace=True)

    # add normalized heat
    heat_df['heat_norm'] = heat_df.heat / heat_df.heat.sum()
    p_0 = heat_df.heat_norm
    #p_0 = features.mean(axis=1)
    beta = 0.3
    W = network / network.sum(axis=0) # normalize A
    assert (np.allclose(W.sum(axis=0), 1)) # assert that rows/cols sum to 1
    p = np.linalg.inv(beta * (np.eye(network.shape[0]) - (1 - beta) * W)).dot(np.array(p_0))
    rwr_ranks = np.argsort(p)[::-1]
    heat_df['rwr_score'] = p
    rwr_pred_test = heat_df[test_mask == 1].rwr_score

    # use MutSigCV -log10 q-values for evaluation of that method
    mutsigcv_scores = pd.read_csv('../data/pancancer/mutsigcv/mutsigcv_genescores.csv',
                                  index_col=0, sep='\t').mean(axis=1)
    nodes = pd.DataFrame(node_names, columns=['ID', 'Name']).set_index('ID')
    mutsigcv_scores_filled = mutsigcv_scores.reindex(nodes.Name).fillna(0)
    mutsigcv_pred_test = mutsigcv_scores_filled[mutsigcv_scores_filled.index.isin(nodes[test_mask].Name)]

    # finally, do the actual plotting
    linewidth = 4
    labelfontsize = 20
    ticksize = 17
    y_true = y_test[test_mask == 1, 0]
    pred_testset_gcn = predictions[predictions.Name.isin(node_names[test_mask, 1])]
    y_true_gcn = pred_testset_gcn.label
    y_score = pred_testset_gcn.Prob_pos
    fpr, tpr, thresholds = roc_curve(y_true=y_true_gcn, y_score=y_score)
    roc_auc = roc_auc_score(y_true=y_true_gcn, y_score=y_score)
    # compute roc for random forest
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_true=y_true, y_score=pred_rf[:, 1])
    roc_auc_rf = roc_auc_score(y_true=y_true, y_score=pred_rf[:, 1])
    # compute ROC for Logistic Regression
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_true=y_true, y_score=pred_lr[:, 1])
    roc_auc_lr = roc_auc_score(y_true=y_true, y_score=pred_lr[:, 1])
    # compute ROC for logistic regression on deepWalk embeddings
    fpr_dw, tpr_dw, thresholds_dw = roc_curve(y_true=y_true, y_score=pred_deepwalk[:, 1])
    roc_auc_dw = roc_auc_score(y_true=y_true, y_score=pred_deepwalk[:, 1])
    # compute ROC for GAT
    fpr_gat, tpr_gat, thresholds_gat = roc_curve(y_true=y_true, y_score=gat_results_test[:, 1])
    roc_auc_gat = roc_auc_score(y_true=y_true, y_score=gat_results_test[:, 1])
    # compute ROC for PageRank
    fpr_pr, tpr_pr, thresholds_pr = roc_curve(y_true=y_true, y_score=pr_pred_test.Score)
    roc_auc_pr = roc_auc_score(y_true=y_true, y_score=pr_pred_test.Score)
    # compute ROC for RWR with HotNet2 heat scores
    fpr_hotnet, tpr_hotnet, thresholds_hotnet = roc_curve(y_true=y_true, y_score=rwr_pred_test)
    roc_auc_hotnet = roc_auc_score(y_true=y_true, y_score=rwr_pred_test)
    # compute ROC for MutSigCV q-values
    fpr_ms, tpr_ms, thresholds_ms = roc_curve(y_true=y_true, y_score=mutsigcv_pred_test)
    roc_auc_ms = roc_auc_score(y_true=y_true, y_score=mutsigcv_pred_test)

    # plot ROC curve
    fig = plt.figure(figsize=(14, 8))
    plt.plot(fpr, tpr, lw=linewidth, label='GCN (AUC = {0:.2f})'.format(roc_auc))
    plt.plot(fpr_rf, tpr_rf, lw=linewidth, label='Rand. Forest (AUC = {0:.2f})'.format(roc_auc_rf))
    #plt.plot(fpr_lr, tpr_lr, lw=linewidth, label='LogReg (AUC = {0:.2f})'.format(roc_auc_lr))
    plt.plot(fpr_dw, tpr_dw, lw=linewidth, label='DeepWalk (AUC = {0:.2f})'.format(roc_auc_dw))
    plt.plot(fpr_gat, tpr_gat, lw=linewidth, label='GAT (AUC = {0:.2f})'.format(roc_auc_gat))
    plt.plot(fpr_pr, tpr_pr, lw=linewidth, label='PageRank (AUC = {0:.2f})'.format(roc_auc_pr))
    plt.plot(fpr_hotnet, tpr_hotnet, lw=linewidth, label='RWR (AUC = {0:.2f})'.format(roc_auc_hotnet))
    plt.plot(fpr_ms[:-1], tpr_ms[:-1], lw=linewidth, label='MutSigCV'.format(roc_auc_ms))
    plt.plot([0, 1], [0, 1], color='gray', lw=linewidth, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=labelfontsize)
    plt.ylabel('True Positive Rate', fontsize=labelfontsize)
    plt.tick_params(axis='both', labelsize=ticksize)
    plt.legend(loc='lower right', prop={'size': 18})
    fig.savefig(os.path.join(model_dir, 'roc_curve.svg'))
    # compute the optimal cutoff according to ROC curve (point on the curve closest to (0, 1))
    distances = np.sqrt(np.sum((np.array([0, 1]) - np.array([fpr, tpr]).T)**2, axis=1))
    idx = np.argmin(distances)
    best_threshold_roc = thresholds[idx]

    linewidth = 4
    labelfontsize = 20
    ticksize = 17
    # calculate precision and recall for GCN
    pr, rec, thresholds = precision_recall_curve(y_true=y_true_gcn, probas_pred=y_score)
    aupr = average_precision_score(y_true=y_true_gcn, y_score=y_score)
    # calculate precision and recall for RF
    pr_rf, rec_rf, thresholds_rf = precision_recall_curve(y_true=y_true, probas_pred=pred_rf[:, 1])
    aupr_rf = average_precision_score(y_true=y_true, y_score=pred_rf[:, 1])
    # calculate precision and recall for Logistic Regression
    pr_lr, rec_lr, thresholds_lr = precision_recall_curve(y_true=y_true, probas_pred=pred_lr[:, 1])
    aupr_lr = average_precision_score(y_true=y_true, y_score=pred_lr[:, 1])
    # calculate precision and recall for logistic regression on deepWalk embeddings
    pr_dw, rec_dw, thresholds_dw = precision_recall_curve(y_true=y_true, probas_pred=pred_deepwalk[:, 1])
    aupr_dw = average_precision_score(y_true=y_true, y_score=pred_deepwalk[:, 1])
    # calculate precision and recall for GAT
    pr_gat, rec_gat, thresholds_gat = precision_recall_curve(y_true=y_true, probas_pred=gat_results_test[:, 1])
    aupr_gat = average_precision_score(y_true=y_true, y_score=gat_results_test[:, 1])
    # calculate precision and recall for PageRank
    pr_pr, rec_pr, thresholds_pr = precision_recall_curve(y_true=y_true, probas_pred=pr_pred_test.Score)
    aupr_pr = average_precision_score(y_true=y_true, y_score=pr_pred_test.Score)
    # calculate precision and recall for Hotnet2
    pr_hotnet, rec_hotnet, thresholds_hotnet = precision_recall_curve(y_true=y_true, probas_pred=rwr_pred_test)
    aupr_hotnet = average_precision_score(y_true=y_true, y_score=rwr_pred_test)
    # compute precision and recall for MutSigCV q-values
    pr_ms, rec_ms, thresholds_ms = precision_recall_curve(y_true=y_true, probas_pred=mutsigcv_pred_test)
    aupr_ms = average_precision_score(y_true=y_true, y_score=mutsigcv_pred_test)

    fig = plt.figure(figsize=(14, 8))
    plt.plot(rec, pr, lw=linewidth, label='GCN (AUPR = {0:.2f})'.format(aupr))
    #plt.plot(rec_svm, pr_svm, lw=linewidth, label='SVM (AUPR = {0:.2f})'.format(aupr_svm))
    plt.plot(rec_rf, pr_rf, lw=linewidth, label='Rand. Forest (AUPR = {0:.2f})'.format(aupr_rf))
    #plt.plot(rec_lr, pr_lr, lw=linewidth, label='LogReg (AUPR = {0:.2f})'.format(aupr_lr))
    plt.plot(rec_dw, pr_dw, lw=linewidth, label='DeepWalk (AUPR = {0:.2f})'.format(aupr_dw))
    plt.plot(rec_gat, pr_gat, lw=linewidth, label='GAT (AUPR = {0:.2f})'.format(aupr_gat))
    plt.plot(rec_pr, pr_pr, lw=linewidth, label='PageRank (AUPR = {0:.2f})'.format(aupr_pr))
    plt.plot(rec_hotnet, pr_hotnet, lw=linewidth, label='HotNet2 (AUPR = {0:.2f})'.format(aupr_hotnet))
    plt.plot(rec_ms[1:], pr_ms[1:], lw=linewidth, label='MutSigCV')
    random_y = y_true.sum() / (y_true.sum() + y_true.shape[0] - y_true.sum())
    plt.plot([0, 1], [random_y, random_y], color='gray', lw=3, linestyle='--', label='Random')
    plt.xlabel('Recall', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.tick_params(axis='both', labelsize=ticksize)
    #plt.title('Precision-Recall Curve')
    plt.legend(prop={'size': 18})
    fig.savefig(os.path.join(model_dir, 'prec_recall.svg'))
    fig.savefig(os.path.join(model_dir, 'prec_recall.png'), dpi=300)
    # compute the optimal cutoff according to PR curve (point closest to (1,1))
    distances = np.sqrt(np.sum((np.array([1, 1]) - np.array([rec, pr]).T)**2, axis=1))
    idx = np.argmin(distances)
    best_threshold_pr = thresholds[idx]
    return best_threshold_roc, best_threshold_pr


def load_predictions(model_dir):
    """Load ensemble predictions and return or calculate it from scratch.

    This function loads the ensemble predictions and re-organizes it to match
    with the index in the HDF5 file.
    If the ensemble prediction was not yet computed, this function will do so.

    Parameters:
    ----------
    model_dir:                  The output directory of the GCN training

    Returns:
    Predictions as pandas DataFrame with 5 columns (Name, label,
    Number of positive predictions, mean probability to be positive and
    standard deviation) for each gene.
    """
    # first, get the data from the container
    args, data_file = gcnIO.load_hyper_params(model_dir)
    if os.path.isdir(data_file): # FIXME: This is hacky and not guaranteed to work at all!
        network_name = None
        for f in os.listdir(data_file):
            if network_name is None:
                network_name = f.split('_')[0].upper()
            else:
                assert (f.split('_')[0].upper() == network_name)
        fname = '{}_{}.h5'.format(network_name, model_dir.strip('/').split('/')[-1])
        data_file = os.path.join(data_file, fname)
    data = gcnIO.load_hdf_data(data_file)
    network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data

    # read predictions, too. Raise error when they are not present
    if not os.path.isfile(os.path.join(model_dir, 'ensemble_predictions.tsv')):
        print ("Ensemble predictions not found. Calculating...")
        compute_ensemble_predictions(model_dir)
    predictions = pd.DataFrame.from_csv(os.path.join(model_dir, 'ensemble_predictions.tsv'),
                                        sep='\t', header=0)
    nodes = pd.DataFrame(node_names, columns=['ID', 'Name']).set_index('ID')
    nodes = nodes[~nodes.index.duplicated()]
    pred_ordered = predictions[~predictions.index.duplicated()]
    pred_ordered.reindex(index=nodes.index)
    predictions.drop([c for c in predictions.columns if c.startswith('Prob_pos')], axis=1, inplace=True)
    predictions.columns = ['Name', 'label', 'Num_Pos', 'Prob_pos', 'Std_Pred']
    return predictions

def compute_overlap(model_dir, fname_out, set1, set2, threshold=0.5, names=['Set1', 'Set2']):
    """Compute the overlap between predictions and other sets.

    This function computes the overlap between the GCN predictions and two other sets and
    plots a Venn diagram of it.

    Parameters:
    ----------
    model_dir:                  The output directory of the GCN training
    fname_out:                  The filename of the resulting plot
                                (will be written to model_dir)
    set1:                       A list, pd.Series or set with probable overlap
    set2:                       Another set that might have overlap with the
                                GCN predictions
    threshold:                  A scalar representing the threshold for the GCN
                                predictions. Default is 0.5
    names:                      Names of set1 and set2 as list of strings.
                                This will be used during plotting.
    """
    predictions = load_predictions(model_dir)

    fig = plt.figure(figsize=(14, 8))
    v = matplotlib_venn.venn3([set(set1),
                            set(predictions[predictions.Prob_pos >= threshold].Name),
                            set(set2)],
            set_labels=[names[0], 'GCN Predictions', names[1]])
    v.get_patch_by_id('10').set_color('#3d3e3d')
    if not v.get_patch_by_id('11') is None:
        v.get_patch_by_id('11').set_color('#37652d')
        v.get_label_by_id('11').set_fontsize(20)
    v.get_patch_by_id('01').set_color('#ee7600')
    v.get_patch_by_id('011').set_color('#4d2600')
    v.get_patch_by_id('001').set_color('#031F6F')
    v.get_label_by_id('A').set_fontsize(20)
    v.get_label_by_id('B').set_fontsize(20)
    v.get_label_by_id('C').set_fontsize(20)
    v.get_label_by_id('10').set_fontsize(20)
    v.get_label_by_id('01').set_fontsize(20)
    if not v.get_patch_by_id('111') is None and not v.get_patch_by_id('101') is None:
        v.get_label_by_id('111').set_fontsize(20)
        v.get_label_by_id('101').set_fontsize(20)
        v.get_patch_by_id('111').set_color('#890707')
        v.get_patch_by_id('101').set_color('#6E80B7')
    v.get_label_by_id('011').set_fontsize(20)
    v.get_label_by_id('001').set_fontsize(20)
    fig.savefig(os.path.join(model_dir, fname_out))


def compute_node_degree_relation(model_dir, threshold=0.5):
    """Plot the relationship between node degree and GCN predictions.

    This function does a scatterplot for each gene with its degree
    (number of neighbors in interaction network) on the x-axis
    and its probability to be a disease gene on the y-axis.
    The color indicates the classification of the gene.
    This function returns the correlation coefficientn between
    the two metrics for all genes.

    Parameters:
    ----------
    model_dir:                  The output directory of the GCN training
    threshold:                  A scalar representing the threshold for the GCN
                                predictions. Default is 0.5

    Returns:
    The correlation coefficient between node degree and probability to be a
    disease gene.
    """
    # get the data from hdf5 container
    args, data_file = gcnIO.load_hyper_params(model_dir)
    if os.path.isdir(data_file): # FIXME: This is hacky and not guaranteed to work at all!
        network_name = None
        for f in os.listdir(data_file):
            if network_name is None:
                network_name = f.split('_')[0].upper()
            else:
                assert (f.split('_')[0].upper() == network_name)
        fname = '{}_{}.h5'.format(network_name, model_dir.strip('/').split('/')[-1])
        data_file = os.path.join(data_file, fname)
    data = gcnIO.load_hdf_data(data_file)
    network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data
    predictions = load_predictions(model_dir)

    node_degree = pd.DataFrame(network.sum(axis=1), index=node_names[:, 0], columns=['Degree'])
    pred_with_degree = node_degree.join(predictions, how='inner', lsuffix='_')

    fig = plt.figure(figsize=(14, 8))
    cmap = plt.cm.BrBG
    plt.scatter(pred_with_degree.Prob_pos, pred_with_degree.Degree,
                c=pred_with_degree.Prob_pos >= threshold,
                cmap=cmap, alpha=.4)
    #plt.gca().set_ylim([0, 600])
    plt.xlabel('GCN Probability', fontsize=20)
    plt.ylabel('Node Degree', fontsize=20)
    plt.tick_params(axis='both', labelsize=17)
    plt.title('Correlation Predictions vs. Node Degree (R={0:.2f})'.format(pred_with_degree.Prob_pos.corr(pred_with_degree.Degree)), fontsize=30)
    pred = mpatches.Patch(color=cmap(1000), label='Predicted Positive')
    non_pred = mpatches.Patch(color=cmap(0), label='Predicted Negative')
    plt.legend(handles=[pred, non_pred], loc='upper left', prop={'size': 18})
    fig.savefig(os.path.join(model_dir, 'degree_correlation.png'), dpi=300)

    return pred_with_degree.Prob_pos.corr(pred_with_degree.Degree)

def parse_args():
    parser = argparse.ArgumentParser(description='Post-process a GCN training by plotting various performance metrics.')
    parser.add_argument('-d', '--dir', help='Training directory',
                        dest='train_dir',
                        required=True,
                        type=str
                        )
    parser.add_argument('-n', '--network', help='PPI network',
                        dest='network_name',
                        required=True,
                        type=str
                        )
    args = parser.parse_args()
    return args


def postprocessing(model_dir, network_name):
    """Run all plotting functions.
    """
    all_preds, all_sets = compute_ensemble_predictions(model_dir)
    compute_node_degree_relation(model_dir, 0.5)
    compute_average_ROC_curve(model_dir, all_preds, all_sets)
    compute_average_PR_curve(model_dir, all_preds, all_sets)
    best_thr_roc, best_thr_pr = compute_ROC_PR_competitors(model_dir, network_name)

    # get the data from hdf5 container
    args, data_file = gcnIO.load_hyper_params(model_dir)
    if os.path.isdir(data_file): # FIXME: This is hacky and not guaranteed to work at all!
        network_name = None
        for f in os.listdir(data_file):
            if network_name is None:
                network_name = f.split('_')[0].upper()
            else:
                assert (f.split('_')[0].upper() == network_name)
        fname = '{}_{}.h5'.format(network_name, model_dir.strip('/').split('/')[-1])
        data_file = os.path.join(data_file, fname)

    data = gcnIO.load_hdf_data(data_file)
    network, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data
    nodes = pd.DataFrame(node_names, columns=['ID', 'Name'])
    nodes['label'] = np.logical_or(np.logical_or(y_train, y_test), y_val)

    # get the NCG cancer genes
    known_cancer_genes = []
    candidate_cancer_genes = []
    n = 0
    with open('../data/pancancer/NCG/cancergenes_list.txt', 'r') as f:
        for line in f.readlines():
            n += 1
            if n == 1:
                continue
            l = line.strip().split('\t')
            if len(l) == 2:
                known_cancer_genes.append(l[0])
                candidate_cancer_genes.append(l[1])
            else:
                candidate_cancer_genes.append(l[0])
    known_cancer_genes_innet = nodes[nodes.Name.isin(known_cancer_genes)].Name
    candidate_cancer_genes_innet = nodes[nodes.Name.isin(candidate_cancer_genes)].Name

    # get blood cancer genes
    cgc = pd.read_csv('../data/pancancer/cosmic/cancer_gene_census.csv')
    cgc.dropna(subset=['Tissue Type'], inplace=True)
    # find blood cancer genes based on these abbreviations (E=Epithelial, M=Mesenchymal, O=Other, L=Leukaemia/lymphoma)
    pattern = '|'.join(['E', 'O', 'M', 'E;'])
    non_blood_cancer_genes = cgc[cgc['Tissue Type'].str.contains(pattern)]
    blood_cancer_genes = cgc[~cgc['Tissue Type'].str.contains(pattern)]
    known_cancer_genes_innet_noblood = non_blood_cancer_genes[non_blood_cancer_genes['Gene Symbol'].isin(known_cancer_genes_innet)]['Gene Symbol']
    known_cancer_genes_innet_blood = blood_cancer_genes[blood_cancer_genes['Gene Symbol'].isin(known_cancer_genes_innet)]['Gene Symbol']

    # compute the Venn diagrams
    compute_overlap(model_dir, 'overlap_NCG.svg',
                    known_cancer_genes_innet, candidate_cancer_genes_innet,
                    best_thr_pr,
                    ['Known Cancer Genes\n(NCG)', 'Candidate Cancer Genes\n(NCG)']
    )
    compute_overlap(model_dir, 'overlap_leukemia_genes.svg',
                               known_cancer_genes_innet_blood, known_cancer_genes_innet_noblood,
                               best_thr_pr,
                               ['Leukemia Genes', 'Non-Leukemia Genes']
                              )

if __name__ == "__main__":
    args = parse_args()
    if not os.path.isfile(os.path.join(args.train_dir, 'hyper_params.txt')):
        print ("Detected no hyper parameter file. Assuming training of all omics separately.")
        for model in os.listdir(args.train_dir):
            model_dir = os.path.join(args.train_dir, model)
            if os.path.isdir(model_dir):
                print ("Running post-processing for {}".format(model_dir))
                postprocessing(model_dir, args.network_name)
    else:
        postprocessing(args.train_dir, args.network_name)