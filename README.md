# Legionella Infection Genes in Host
This project aims at identifying new host genes that are involved in the infection of legionella pneunophila.
In order to do that, different methods are used that combine protein-protein interaction networks and gene expression
data obtained from RNA-seq experiments.

## Installation
Requires the following python packages:
* Tensorflow
* goatools
* mygene
* ...

## Methods
In the project, we assume that genes relevant for the infection of legionella are located in proximity in the network.
Further, we assume that the dual RNA-seq data identifies putative infection genes in the host, too.

This is why we tested different algorithms, combining the two data sources:
* **NetRank**, an algorithm that assigns scores to nodes and ranks all nodes based on the scores and the network properties.
This algorithm is based on the famous PageRank algorithm and was described in "Google Goes Cancer".
* **Graph Convolutional Networks**, a sophisticated neural network approach which uses the concepts of Convolutional Neural Networks
and spectral graph theory. It learns to predict new nodes in the network by combining the local topology around a node in question
as well as the node properties of that neighborhood. The algorithm can deal with vectors for the node scores and also incorporate
time series data. It was described by Kipf & Welling in "Semi-Supervised Classification with Graph Convolutional Networks".

## Usage
TBD

## Results
TBD
