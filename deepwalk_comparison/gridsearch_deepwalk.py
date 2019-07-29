import argparse
import os, sys, subprocess
import pickle

import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
import gcnIO
import gcnPreprocessing
import utils


def write_hyper_param_dict(params, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)

def load_hyper_param_dict(file_name):
    with open(file_name, 'rb') as f:
        params = pickle.load(f)
    return params

def write_performances(performance, file_name):
    write_hyper_param_dict(performance, file_name)

def load_performances(file_name):
    return load_hyper_param_dict(file_name)

def check_param_already_done(training_dir, params):
    for setting in os.listdir(training_dir):
        if setting.startswith('params'):
            other_params = load_hyper_param_dict(os.path.join(training_dir, setting, 'params.txt'))
            if other_params == params:
                return load_performances(os.path.join(training_dir, setting, 'performance.txt'))
    return None


def run_model(params, out_file, n_jobs=1):
    cmd = ("deepwalk --format edgelist "
           "--input /project/gcn/diseasegcn/data/networks/CPDB_numeric.csv "
           "--number-walks {} "
           "--representation-size {} --walk-length {} --window-size {} "
           "--workers {} --output {}"
    )
    cmd_completed = cmd.format(params['number_of_walks'],
                               params['representation_size'],
                               params['walk_length'],
                               params['window_size'],
                               n_jobs, out_file)
    subprocess.call(cmd_completed, shell=True)


def get_model_performance(embedding_file, features, node_names, y_train, train_mask, y_test, test_mask):
    deepwalk_embeddings = pd.read_csv(embedding_file, sep='\t')
    nodes = pd.DataFrame(node_names, columns=['ID', 'Name']).set_index('Name')
    X_dw = deepwalk_embeddings.set_index('Name').reindex(nodes.index).drop('Node_Id', axis=1)
    X_train_dw = X_dw[train_mask.astype(np.bool)]
    X_test_dw = X_dw[test_mask.astype(np.bool)]
    log_reg_dw = LogisticRegression(class_weight='balanced')
    log_reg_dw.fit(X_train_dw, y_train.reshape(-1))
    pred_deepwalk = log_reg_dw.predict_proba(X_test_dw)
    pr_dw, rec_dw, thresholds_dw = precision_recall_curve(y_true=y_true,
                                                          probas_pred=pred_deepwalk[:, 1])
    aupr_dw = average_precision_score(y_true=y_true, y_score=pred_deepwalk[:, 1])
    fig = plt.figure(figsize=(14, 8))
    plt.plot(rec_dw, pr_dw, lw=linewidth, label='DeepWalk (AUPR = {0:.2f})'.format(aupr_dw))
    plt.legend()
    print (os.path.dirname(embedding_file))
    fig.savefig(os.path.join(os.path.dirname(embedding_file), 'pr_curve.svg'))
    return aupr_dw

def parse_args():
    parser = argparse.ArgumentParser(description='Grid search for deepWalk')
    parser.add_argument('-d', '--data', help='Path to the data container',
                        dest='data',
                        type=str,
                        required=True
                        )
    parser.add_argument('-n', '--n_jobs', help='Number of jobs to run in parallel',
                        dest='n_jobs',
                        type=int,
                        default=1
                        )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print ("Loading Data...")
    data = gcnIO.load_hdf_data(args.data,
                               feature_name='features')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feat_names = data
    num_nodes = adj.shape[0]

    params = {'number_of_walks':[10, 40, 80],
              'representation_size':[64, 128, 256],
              'walk_length': [40, 80, 120],
              'window_size':[10]
              }
    """
    params = {'number_of_walks':[10, 80],
              'representation_size':[256],
              'walk_length': [40],
              'window_size':[10]
              }
    """
    num_of_settings = len(list(ParameterGrid(params)))
    print ("Grid Search: Trying {} different parameter settings...".format(num_of_settings))
    param_num = 0
    # create output directory or set existing one
    out_dir = '/project/gcn/deepwalk/deepwalk_gridsearch'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # create session, train and save afterwards
    performances = []
    for param_set in list(ParameterGrid(params)):
        run_model(params=param_set,
                  out_file=os.path.join(out_dir, 'embedding_{}'.format(param_num)),
                  n_jobs=args.n_jobs)
        param_num += 1
        write_hyper_param_dict(param_set, os.path.join(out_dir, 'params_{}.txt'.format(param_num)))
        print ("[{} out of {} combinations]: {}".format(param_num, num_of_settings, param_set))
    # write results from gridsearch to file
    out_name = os.path.join(out_dir, 'final_results.pkl')
    with open(out_name, 'wb') as f:
        pickle.dump(performances, f)
