# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:49:32 2017

@author: roman
"""
import numpy as np
import pagerank
import argparse, os

def netrank_gridsearch(network_path, diff_expr, out_path, alpha_prec=10):
    """Perform grid search over alpha parameter.
    
    This function will compute the netranks for a given network and differential
    expression data for a range of alpha parameters.
    
    Parameters:
    ----------
    network_path:           Path to the network in hdf5 container (with gene names).
                            Gene names and expression data are assumed to be in
                            the same order as the nodes in the adjacency matrix.

    diff_expr:              Differential expression dataframe. If set to None,
                            PageRank is calculated, otherwise NetRank will be used.

    out_path:               Directory to which the results are written.

    alpha_prec:             The number of runs to compute netrank scores.
                            Default is 10, which corresponds to computing ranks
                            for alpha = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1].
    """
    alpha_range = np.linspace(0, 1, alpha_prec)
    for alpha in alpha_range:
        scores, gene_names = pagerank.pagerank(network_path, diff_expr, alpha)
        out = os.path.join(out_path, 'netrank_alpha_{}.txt'.format(alpha))
        pagerank.write_ranking(scores, gene_names, out)
        print ("Netrank for alpha {} computed successfully!".format(alpha))
    print ("Grid Search successfully computed (results in {})".format(out_path))


def parseArgs():
    parser = argparse.ArgumentParser(description='Execute PageRank/NetRank on PPI Network')
    parser.add_argument('--ppi', help='path to ppi network hdf5 container',
                        dest='ppi'
                        )
    parser.add_argument('--de_up',
                        help='differential expression up-regulated (html)',
                        dest='de_up',
                        default=None,
                        type=str
                        )
    parser.add_argument('--de_down',
                        help='differential expression down-regulated (html)',
                        dest='de_down',
                        default=None,
                        type=str
                        )
    parser.add_argument('--out',
                        help='Path to output directory',
                        dest='out_path',
                        type=str
                        )
    args = parser.parse_args()
    return args.ppi, args.de_up, args.de_down, args.out_path


if __name__ == "__main__":
    ppi_path, de_up_path, de_down_path, out_path = parseArgs()

    # load DE (usually GFP+ vs. Control after 16 hours for up and down-regulated genes)
    # Unfortunately, we only have pvalue < .05, fill the rest with zeros.
    de = pagerank.load_diff_expr(de_up_path, de_down_path)
    netrank_gridsearch(ppi_path, de)