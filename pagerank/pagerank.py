import networkx as nx
import numpy as np
import h5py
import argparse
import operator
import pandas

if __name__ == "__main__":
    # load network and gene expression data
    fname = '../data/preprocessing/ppi_networks.h5'
    with h5py.File(fname, 'r') as f:
        gene_expression_data = f['gene_expression'][:]
        ppi_network = f['consensusPathDB_ppi'][:]
        gene_names = f['gene_names'][:]

        # build networkx graph and do pagerank on it
        G = nx.from_numpy_matrix(ppi_network)
        pagerank_vals = nx.pagerank(G)
        maxi = max(pagerank_vals, key=pagerank_vals.get)
        print ("Maximum Pagerank: Index: {}\tPagerank: {}".format(maxi, pagerank_vals[maxi]))
        mini = min(pagerank_vals, key=pagerank_vals.get)
        print ("Minimum Pagerank: Index: {}\tPagerank: {}".format(mini, pagerank_vals[mini]))

        # sort pagerank results (dict with gene idx and pagerank val)
        pagerank_sorted = sorted(pagerank_vals.items(), key=operator.itemgetter(1))[::-1]

        # get the gene names and print to file
        result_file = '../data/pagerank/pagerank_scores.txt'
        with open(result_file, 'w') as res:
            count = 1
            res.write('Gene_ID\tGene_Name\tRank\tPageRank_Score\n')
            for gene_idx, pr in pagerank_sorted:
                res.write("{}\t{}\t{}\t{}\n".format(gene_names[gene_idx][0], gene_names[gene_idx][1], count, pr))
                count += 1
        print ("Done with PageRank Algorithm! Exit!")
