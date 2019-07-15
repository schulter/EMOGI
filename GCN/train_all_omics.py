# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:29:41 2018

@author: roman
"""

# data wrangling
import argparse, os, sys
import utils, gcnIO, gcnPreprocessing

# computation
import tensorflow as tf
from scipy.sparse import lil_matrix
import scipy.sparse as sp
import numpy as np
from train_gcn import *
from my_gcn import MYGCN
from train_cv import run_all_cvs

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import math
bestSplit = lambda x: (round(math.sqrt(x)), math.ceil(x / round(math.sqrt(x))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    parser.add_argument('-hd', '--hidden_dims',
                        help='Hidden Dimensions (number of filters per layer. Also determines the number of hidden layers.',
                        nargs='+',
                        dest='hidden_dims',
                        required=True)
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
    parser.add_argument('-d', '--data', help='Path to folder containing the data containers',
                        dest='data',
                        type=str
                        )
    parser.add_argument('-cv', '--cv_runs', help='Number of cross validation runs',
                    dest='cv_runs',
                    default=10,
                    type=int
                    )
    args = parser.parse_args()
    return args


def basic_plots(performance_dicts, names, outfile):
    r, c = bestSplit(len(performance_dicts))
    print (r, c)
    fig, ax = plt.subplots(nrows=r, ncols=c, figsize=(30, 20), squeeze=False)
    number_of_plot = 0
    for i in performance_dicts:
        x = ['loss', 'acc', 'aupr', 'auroc']
        height = [np.mean(i[metric]) for metric in x]
        err = [np.std(i[metric]) for metric in x]
        ax[number_of_plot // c][number_of_plot % c].set_title(names[number_of_plot], fontsize=35)
        ax[number_of_plot // c][number_of_plot % c].bar(x, height, color="#5ab4ac",
                  yerr=err)
        for tick in ax[number_of_plot // c][number_of_plot % c].xaxis.get_major_ticks():
            tick.label.set_fontsize(30)
        for tick in ax[number_of_plot // c][number_of_plot % c].yaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        number_of_plot += 1
    fig.savefig(outfile)


def get_performance_dict(performance_measures):
    performance_dict = {'loss': [], 'acc': [], 'aupr': [], 'auroc': []}
    for p in performance_measures:
        performance_dict['loss'].append(p[0])
        performance_dict['acc'].append(p[1])
        performance_dict['aupr'].append(p[2])
        performance_dict['auroc'].append(p[3])
    return performance_dict

if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    hidden_dims = [int(x) for x in args.hidden_dims]

    if not os.path.isdir(args.data):
        print("Data is not a directory. This script uses all HDF5 containers in the directory for training. Consider using `train_cv.py` for training only one HDF5 container.")
        sys.exit(-1)

    output_dir = gcnIO.create_model_dir()

    # run all omics datasets that we could find.
    all_omics_p = []
    data_types = []
    base_dir = args.data
    for data_file in os.listdir(base_dir):
        if data_file.endswith('.h5'):
            print ("Detected {}. Running GCN with that file!".format(data_file))
            omics_type = '_'.join(data_file.split('.')[0].split('_')[1:])
            data_types.append(omics_type)
            training_dir = os.path.join(output_dir, omics_type)

            # load data and preprocess it
            input_data_path = os.path.join(base_dir, data_file)
            data = gcnIO.load_hdf_data(input_data_path, feature_name='features')
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data
            print("Read data from: {}".format(input_data_path))

            args_dict['data'] = input_data_path
            performance_measures = run_all_cvs(adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names, args_dict, training_dir)
            all_omics_p.append(get_performance_dict(performance_measures))
    basic_plots(all_omics_p, data_types, os.path.join(output_dir, 'performance_statistics.svg'))