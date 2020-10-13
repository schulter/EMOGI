import pandas as pd
import numpy as np
import mygene
import networkx as nx
import os, sys, h5py

def get_symbols_from_ensembl(list_of_ensembl_ids):
    """Get the hugo gene symbols from a list of Ensembl IDs using mygene.

    This function retrieves hugo gene symbols from Ensembl IDs using
    the mygene python API. It requires a stable internet connection to
    retrieve the annotations.
    @see get_ensembl_from_symbol

    Parameters:
    ----------
    list_of_ensembl_ids:        A list of strings containing the
                                Ensembl IDs
    
    Returns:
    A dataframe containing the mapping between symbols and ensembl IDs.
    If no symbol could be found for an ID, NA is returned in that row.
    The index of the dataframe are the ensembl IDs and the symbols are
    in the other column.
    """
    # get Ensembl IDs for gene names
    mg = mygene.MyGeneInfo()
    res = mg.querymany(list_of_ensembl_ids,
                       scopes='ensembl.gene',
                       fields='symbol',
                       species='human', returnall=True
                      )

    def get_symbol_and_ensembl(d):
        if 'symbol' in d:
            return [d['query'], d['symbol']]
        else:
            return [d['query'], None]

    node_names = [get_symbol_and_ensembl(d) for d in res['out']]
    # now, retrieve the names and IDs from a dictionary and put in DF
    node_names = pd.DataFrame(node_names, columns=['Ensembl_ID', 'Symbol']).set_index('Ensembl_ID')
    node_names.dropna(axis=0, inplace=True)
    return node_names

def get_ensembl_from_symbol(list_of_gene_symbols):
    """Get the Ensembl IDs from a list of hugo gene symbols using mygene.

    This function retrieves Ensembl IDs from hugo gene symbols using
    the mygene python API. It requires a stable internet connection to
    retrieve the annotations.
    @see get_symbols_from_ensembl

    Parameters:
    ----------
    list_of_gene_symbols:       A list of strings containing the
                                hugo gene symbols
    
    Returns:
    A dataframe containing the mapping between symbols and ensembl IDs.
    If no symbol could be found for an ID, NA is returned in that row.
    The index of the dataframe are the symbols and the ensembl IDs are
    in the other column.
    """
    # get Ensembl IDs for gene names
    mg = mygene.MyGeneInfo()
    res = mg.querymany(list_of_gene_symbols,
                       scopes='symbol, refseq, uniprot',
                       fields='ensembl.gene',
                       species='human', returnall=True
                      )

    # now, retrieve the names and IDs from a dictionary and put in DF
    def get_name_and_id(x):
        if 'ensembl' in x:
            ens_id = x['ensembl'][0]['gene'] if type(x['ensembl']) is list else x['ensembl']['gene']
            symbol = x['query']
            return [symbol, ens_id]
        else:
            return [x['query'], None]

    ens_ids = [get_name_and_id(x) for x in res['out']]
    
    node_names = pd.DataFrame(ens_ids, columns=['Symbol', 'Ensembl_ID']).set_index('Ensembl_ID')
    node_names.dropna(axis=0, inplace=True)
    node_names.drop_duplicates(inplace=True)
    return node_names


def get_entrez_from_symbol(list_of_symbols):
    """Get the entrez IDs from a list of Hugo Gene Symbols using mygene.

    This function retrieves Entrez IDs from Hugo Gene Symbols using
    the mygene python API. It requires a stable internet connection to
    retrieve the annotations.
    @see get_ensembl_from_symbol

    Parameters:
    ----------
    list_of_symbols:        A list of strings containing the
                                Ensembl IDs
    
    Returns:
    A dataframe containing the mapping between symbols and Entrez IDs.
    If no symbol could be found for an ID, NA is returned in that row.
    The index of the dataframe are the Entrez IDs and the symbols are
    in the other column.
    """
    # get Ensembl IDs for gene names
    mg = mygene.MyGeneInfo()
    res = mg.querymany(list_of_symbols,
                       scopes='symbol',
                       fields='entrezgene',
                       species='human', returnall=True
                      )

    def get_symbol_and_entrez(x):
        if 'entrezgene' in x:
            entrez_id = x['entrezgene']
            symbol = x['query']
            return [symbol, entrez_id]
        else:
            return [x['query'], None]
    output = [get_symbol_and_entrez(d) for d in res['out']]
    # now, retrieve the names and IDs from a dictionary and put in DF
    output = pd.DataFrame(output, columns=['Symbol', 'Entrez_ID']).set_index('Entrez_ID')
    output.dropna(axis=0, inplace=True)
    return output


def load_PPI_network(network_name, verbose=False):
    """Load one of several PPI networks which we use in the analysis.

    This function loads a requested PPI network as an adjacency matrix.
    The network is returned as a pandas DataFrame with columns and index being
    the the gene symbols. 1 indicates that there is an edge between two genes
    (or rather the proteins of that gene) and zero indicates that there is no edge
    between the two.

    Parameters:
    ----------
    network_name:               The name of the network we use. Can be any of
                                "IREF" (for the IRefIndex PPI network), "CPDB",
                                "Multinet", "STRING" (for STRING-db), "PCNet"
                                or "IREFNew" (for the current version of IRefIndex).
    verbose:                    True if some output on the number of edges and nodes
                                shall be written to stdout. Defaults to False.
    
    Returns:
    A pandas DataFrame containing the adjacency matrix of the network. Columns and
    index are identical and correspond to the gene symbols.
    If the network_name does not correspond to any of the valid networks, None is
    returned and an error is written to stderr.
    """
    if network_name.upper() == 'IREF':
        net_file = '../../data/pancancer/hotnet2/networks/irefindex9/irefindex9_edge_list'
        name_file = '../../data/pancancer/hotnet2/networks/irefindex9/irefindex9_index_gene'
        edgelist = pd.read_csv(net_file, sep=' ', header=None,
                            names=['from', 'to', 'weight'])
        index = pd.read_csv(name_file, sep=' ', header=None, names=['name'])
        # build network and relabel nodes to match with real names
        ppi_graph = nx.from_pandas_edgelist(edgelist, source='from', target='to', edge_attr=None)
        _ = nx.relabel_nodes(ppi_graph, index.to_dict()['name'], copy=False)
        ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
        symbols_network_genes = get_ensembl_from_symbol(ppi_network.index)
        # Remove nodes from network that don't have corresponding gene names
        nodes_not_translatable = ppi_network[~ppi_network.index.isin(symbols_network_genes.Symbol)].index
        print ("Not translatable Ensembl IDs: {}".format(nodes_not_translatable.shape[0]))
        ppi_graph.remove_nodes_from(nodes_not_translatable)
        ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
        assert ((ppi_network.index == symbols_network_genes.Symbol).all())
    elif network_name.upper() == 'MULTINET':
        net_file = '../../data/pancancer/hotnet2/networks/multinet/multinet_edge_list'
        name_file = '../../data/pancancer/hotnet2/networks/multinet/multinet_index_gene'
        edgelist = pd.read_csv(net_file, sep=' ', header=None,
                            names=['from', 'to', 'weight'])
        index = pd.read_csv(name_file, sep=' ', header=None, names=['name'])
        # build network and relabel nodes to match with real names
        ppi_graph = nx.from_pandas_edgelist(edgelist, source='from', target='to', edge_attr=None)
        _ = nx.relabel_nodes(ppi_graph, index.to_dict()['name'], copy=False)
        ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
        symbols_network_genes = get_ensembl_from_symbol(ppi_network.index)
        # Remove nodes from network that don't have corresponding gene names
        nodes_not_translatable = ppi_network[~ppi_network.index.isin(symbols_network_genes.Symbol)].index
        if verbose:
            print ("Not translatable Ensembl IDs: {}".format(nodes_not_translatable.shape[0]))
        ppi_graph.remove_nodes_from(nodes_not_translatable)
        ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
    elif network_name.upper() == 'CPDB':
        ppi_network = pd.read_csv('../../data/networks/CPDB_symbols_edgelist.tsv', sep='\t')
        ppi_graph = nx.from_pandas_edgelist(df=ppi_network, source='partner1', target='partner2', edge_attr='confidence')
        ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
    elif network_name.upper() == 'STRING':
        ppi_network = pd.read_csv('../../data/networks/string_SYMBOLS_highconf.tsv', sep='\t', compression='gzip')
        ppi_graph = nx.from_pandas_edgelist(df=ppi_network, source='partner1', target='partner2', edge_attr='confidence')
        ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
    elif network_name.upper() == 'IREF_NEW':
        ppi_network = pd.read_csv('../../data/networks/IREF_symbols_20190730.tsv', sep='\t', compression='gzip')
        ppi_graph = nx.from_pandas_edgelist(df=ppi_network, source='partner1', target='partner2', edge_attr='confidence')
        ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
    elif network_name.upper() == 'PCNET':
        ppi_network = pd.read_csv('../../data/networks/pcnet_edgelist.tsv.gz', sep='\t', compression='gzip')
        ppi_graph = nx.from_pandas_edgelist(df=ppi_network, source='Source_Name', target='Target_Name')
        ppi_network = nx.to_pandas_adjacency(G=ppi_graph)
    else:
        print ("No PPI network named {}".format(network_name), file=sys.stderr)

    if verbose:
        print ("Edges: {}\tNodes: {}".format(ppi_graph.number_of_edges(), ppi_graph.number_of_nodes()))

    return ppi_network


def get_positive_labels(nodes, strategy='all', cancer_type='pancancer', remove_blood_cancer_genes=False, verbose=False):
    """Compute positive labels (known cancer genes) for the GCN using one of different strategies.

    This function loads known cancer genes from different databases, according to the chosen
    strategy. Similar to the other util functions, it relies on the data being stored in
    the correct locations.

    Parameters:
    nodes:                      A DataFrame with the nodes of a PPI network. This is considered
                                the universe from which we can load the cancer genes.
                                If a gene is in one of the databases but not in the nodes, it
                                will be discarded.
                                Has to contain a column called "Name" with the hugo gene symbols
                                of the network in question.
    strategy:                   How to select the known cancer genes. Can be any of "NCG", "Expression",
                                "Methylation", "Mutation" or "all". The NCG is a database of known
                                and candidate cancer genes from which only the known ones are retrieved.
                                Mutation, Methylation and Expression refer to the literature mining
                                tool DigSEE which collects cancer genes based on evidence type.
    remove_blood_cancer_genes:  If genes that are associated with (only) leukemias are removed
                                from the list or not. The annotation for cancer genes is derived
                                from the COSMIC CGC.
    verbose:                    Whether or ont to output information on the selection process
                                and how many cancer genes have been retrieved.
    
    Returns:
    A subset of the "nodes" DataFrame with the cancer genes.
    """
    os_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    ncg_known_cancer_genes = []
    n = 0
    with open('../../data/pancancer/NCG/cancergenes_list.txt', 'r') as f:
        for line in f.readlines():
            n += 1
            if n == 1:
                continue
            l = line.strip().split('\t')
            if len(l) == 2:
                ncg_known_cancer_genes.append(l[0])

    if strategy.upper() == 'NCG':
        known_cancer_genes_innet = nodes[nodes.Name.isin(ncg_known_cancer_genes)].Name
        print (known_cancer_genes_innet.shape, len(ncg_known_cancer_genes))

    elif strategy.upper() == 'EXPRESSION':
        known_cancer_genes = []
        if cancer_type.upper() == 'PANCANCER':
            with open('../../data/pancancer/digSEE/expression/pan_cancer_genes.txt', 'r') as f:
                for line in f.readlines():
                    known_cancer_genes.append(line.strip())
        else:
            evidence = pd.read_csv('../../data/pancancer/digSEE/expression/expression_{}.txt'.format(cancer_type.upper()), sep='\t')
            high_scores = evidence[evidence['EVIDENCE SENTENCE SCORE'] >= 0.3]
            known_cancer_genes = high_scores['GENE SYMBOL'].tolist()
        known_cancer_genes_innet = nodes[nodes.Name.isin(known_cancer_genes)].Name

    elif strategy.upper() == 'METHYLATION':
        known_cancer_genes = []
        if cancer_type.upper() == 'PANCANCER':
            with open('../../data/pancancer/digSEE/methylation/pan_cancer_genes.txt', 'r') as f:
                for line in f.readlines():
                    known_cancer_genes.append(line.strip())
        else:
            evidence = pd.read_csv('../../data/pancancer/digSEE/methylation/methylation_{}.txt'.format(cancer_type.upper()), sep='\t')
            high_scores = evidence[evidence['EVIDENCE SENTENCE SCORE'] >= 0.3]
            known_cancer_genes = high_scores['GENE SYMBOL'].tolist()
        known_cancer_genes_innet = nodes[nodes.Name.isin(known_cancer_genes)].Name

    elif strategy.upper() == 'MUTATION':
        known_cancer_genes = []
        if cancer_type.upper() == 'PANCANCER':
            with open('../../data/pancancer/digSEE/mutation/pan_cancer_genes.txt', 'r') as f:
                for line in f.readlines():
                    known_cancer_genes.append(line.strip())
        else:
            evidence = pd.read_csv('../../data/pancancer/digSEE/mutation/mutation_{}.txt'.format(cancer_type.upper()), sep='\t')
            high_scores = evidence[evidence['EVIDENCE SENTENCE SCORE'] >= 0.3]
            known_cancer_genes = high_scores['GENE SYMBOL'].tolist()
        known_cancer_genes_innet = nodes[nodes.Name.isin(known_cancer_genes)].Name

    elif strategy.upper() == 'ALL':
        # start with NCG genes
        known_cancer_genes = ncg_known_cancer_genes
        if cancer_type.upper() == 'PANCANCER':
            # add expression literature evidence (digSEE)
            with open('../../data/pancancer/digSEE/expression/pan_cancer_genes.txt', 'r') as f:
                for line in f.readlines():
                    known_cancer_genes.append(line.strip())
            # add mutational literature evidence (digSEE)
            with open('../../data/pancancer/digSEE/mutation/pan_cancer_genes.txt', 'r') as f:
                for line in f.readlines():
                    known_cancer_genes.append(line.strip())
            # add methylation literature evidence (digSEE)
            with open('../../data/pancancer/digSEE/methylation/pan_cancer_genes.txt', 'r') as f:
                for line in f.readlines():
                    known_cancer_genes.append(line.strip())
        else:
            # start with COSMIC genes that contain the tissue in question
            tissue_pairs = {'blca': ['bladder', 'urinary'],
                            'brca': ['breast'],
                            'cesc': ['cervix', 'cervical', 'endocervical'],
                            'ucec': ['uterus', 'uterine'],
                            'coad': ['colon', 'colorectal'],
                            'lihc': ['liver'],
                            'hnsc': ['salivary', 'head and neck'],
                            'esca': ['esophagus', 'esophageal'],
                            'prad': ['prostate'],
                            'stad': ['stomach'],
                            'thca': ['thyroid'],
                            'lusc': ['lung'],
                            'kirp': ['kidney']
                           }
            cosmic_gene_scores = pd.read_csv('../../data/pancancer/cosmic/cancer_gene_census.csv', header=0)
            cosmic_gene_scores.dropna(subset=['Tumour Types(Somatic)'], axis=0, inplace=True)
            genes_for_ctype = cosmic_gene_scores[cosmic_gene_scores['Tumour Types(Somatic)'].str.contains('|'.join(tissue_pairs[cancer_type.lower()]))]
            known_cancer_genes = list(genes_for_ctype['Gene Symbol'])
            if verbose:
                print ("Got {} COSMIC CGC Genes".format(len(known_cancer_genes)))

            # add digSEE genes for the cancer type in question for all omics
            for evidence_type in ['mutation', 'methylation', 'expression']:
                fname = '../../data/pancancer/digSEE/{0}/{0}_{1}.txt'.format(evidence_type, cancer_type.upper())
                evidence = pd.read_csv(fname, sep='\t')
                high_scores = evidence[evidence['EVIDENCE SENTENCE SCORE'] >= 0.8]
                known_cancer_genes += high_scores['GENE SYMBOL'].tolist()
                if verbose:
                    print ("Added {} DigSEE {} Genes".format(high_scores['GENE SYMBOL'].nunique(), evidence_type))
            known_cancer_genes = list(set(known_cancer_genes)) # remove duplicates
        known_cancer_genes_innet = nodes[nodes.Name.isin(known_cancer_genes)].Name
    else:
        print ("Label Source {} not understood.".format(strategy), file=sys.stderr)
    
    if verbose:
        print ("Collected {} cancer genes with strategy {}".format(len(known_cancer_genes_innet), strategy))

    if remove_blood_cancer_genes:
        # load cgc
        cgc = pd.read_csv('../../data/pancancer/cosmic/cancer_gene_census.csv')
        cgc.dropna(subset=['Tissue Type'], inplace=True)

        # find blood cancer genes based on these abbreviations (E=Epithelial, M=Mesenchymal, O=Other, L=Leukaemia/lymphoma)
        pattern = '|'.join(['E', 'O', 'M', 'E;'])
        non_blood_cancer_genes = cgc[cgc['Tissue Type'].str.contains(pattern)]
        known_cancer_genes_innet = non_blood_cancer_genes[non_blood_cancer_genes['Gene Symbol'].isin(known_cancer_genes_innet)]['Gene Symbol']
        if verbose:
            print ("Left with {} known cancer genes after removal of blood cancer genes".format(known_cancer_genes_innet.shape[0]))
    os.chdir(os_cwd)
    return known_cancer_genes_innet


def get_negative_labels(nodes, positives, ppi_network, min_degree=1, verbose=False):
    if verbose:
        print ("{} genes are in network".format(nodes.shape[0]))
    # get rid of the positives (known cancer genes)
    not_positives = nodes[~nodes.Name.isin(positives)]
    if verbose:
        print ("{} genes are in network but not in positives (known cancer genes from NCG)".format(not_positives.shape[0]))

    # get rid of OMIM genes associated with cancer
    omim_cancer_genes = pd.read_csv('../../data/pancancer/OMIM/genemap_search_cancer.txt',
                                    sep='\t', comment='#', header=0, skiprows=3)
    # use fact that nan != nan for filtering out NaN
    sublists = [sublist for sublist in omim_cancer_genes['Gene/Locus'].str.split(',') if sublist == sublist]
    omim_cancer_geneset = [item.strip() for sublist in sublists for item in sublist]
    not_omim_not_pos = not_positives[~not_positives.Name.isin(omim_cancer_geneset)]
    if verbose:
        print ("{} genes are also not in OMIM cancer genes".format(not_omim_not_pos.shape[0]))

    # get rid of all the OMIM disease genes
    omim_genes = pd.read_csv('../../data/pancancer/OMIM/genemap2.txt', sep='\t', comment='#', header=None)
    omim_genes.columns = ['Chromosome', 'Genomic Position Start', 'Genomic Position End', 'Cyto Location',
                        'Computed Cyto Location', 'Mim Number', 'Gene Symbol', 'Gene Name',
                        'Approved Symbol', 'Entrez Gene ID', 'Ensembl Gene ID', 'Comments',
                        'Phenotypes', 'Mouse Gene Symbol/ID']
    omim_gene_names = []
    for idx, row in omim_genes.iterrows():
        gene_names = row['Gene Symbol'].strip().split(',')
        omim_gene_names += gene_names
    omim_gene_names = list(set(omim_gene_names))
    not_omim_not_pos = not_omim_not_pos[~not_omim_not_pos.Name.isin(omim_gene_names)]
    if verbose:
        print ("{} genes are in network but not in oncogenes and not in OMIM".format(not_omim_not_pos.shape[0]))

    # remove COSMIC cancer gene census genes
    cosmic_gene_scores = pd.read_csv('../../data/pancancer/cosmic/cancer_gene_census.csv', header=0)
    not_omim_cosmic_pos = not_omim_not_pos[~not_omim_not_pos.Name.isin(cosmic_gene_scores['Gene Symbol'])]
    if verbose:
        print ("{} genes are also not in COSMIC cancer gene census".format(not_omim_cosmic_pos.shape[0]))

    # remove COSMIC highly mutated genes
    cosmic_prcoding_mutations = pd.read_csv('../../data/pancancer/cosmic/CosmicMutantExportCensus.tsv.gz',
                                            compression='gzip', sep='\t')
    non_pos_omim_cosmiccgc_cosmic_mutated = not_omim_cosmic_pos[~not_omim_cosmic_pos.Name.isin(cosmic_prcoding_mutations['Gene name'])]
    if verbose:
        print ("{} genes are also not in COSMIC mutated genes".format(non_pos_omim_cosmiccgc_cosmic_mutated.shape[0]))

    # remove genes that belong to KEGG pathways in cancer
    kegg_cancer_pathway_genes = pd.read_csv('../../data/pancancer/KEGG/KEGG_genes_in_pathways_in_cancer.txt',
                                            skiprows=2, header=None, names=['Name'])
    not_pos_omim_cosmic_kegg = non_pos_omim_cosmiccgc_cosmic_mutated[~non_pos_omim_cosmiccgc_cosmic_mutated.Name.isin(kegg_cancer_pathway_genes.Name)]
    if verbose:
        print ("{} genes are also not in KEGG cancer pathways".format(not_pos_omim_cosmic_kegg.shape[0]))

    # get rid of genes that are not candidate cancer genes
    ncg_candidate_cancer_genes = []
    n = 0
    with open('../../data/pancancer/NCG/cancergenes_list.txt', 'r') as f:
        for line in f.readlines():
            n += 1
            if n == 1:
                continue
            l = line.strip().split('\t')
            if len(l) == 2:
                ncg_candidate_cancer_genes.append(l[1])
            else:
                ncg_candidate_cancer_genes.append(l[0])
    negatives = not_pos_omim_cosmic_kegg[~not_pos_omim_cosmic_kegg.Name.isin(ncg_candidate_cancer_genes)]
    if verbose:
        print ("{} genes are also not in NCG candidate cancer genes".format(negatives.shape[0]))

    """
    # collect genes in KEGG cancer modules
    kegg_cancer_module_genes = []
    count = 0
    with open('../../data/pancancer/KEGG/KEGG_cancer_modules.gmt', 'r') as f:
        for line in f:
            for item in line.split('\t')[2:]:
                kegg_cancer_module_genes.append(item.strip())
    kegg_cancer_module_genes = list(set(kegg_cancer_module_genes))

    # collect genes in KEGG cancer gene neighborhoods
    kegg_cancer_neighborhood_genes = []
    count = 0
    with open('../../data/pancancer/KEGG/KEGG_cancer_neighborhoods.gmt', 'r') as f:
        for line in f:
            for item in line.split('\t')[2:]:
                kegg_cancer_neighborhood_genes.append(item.strip())
    kegg_cancer_neighborhood_genes = list(set(kegg_cancer_neighborhood_genes))

    negatives = negatives[~negatives.Name.isin(kegg_cancer_module_genes) & ~negatives.Name.isin(kegg_cancer_neighborhood_genes)]
    print ("{} genes also not in KEGG cancer modules or KEGG cancer gene neighborhoods".format(negatives.shape[0]))
    """

    # remove very low degree genes to lower the bias
    degrees_with_labels = pd.DataFrame(ppi_network.sum(), columns=['Degree'])
    neg_w_degrees = degrees_with_labels[degrees_with_labels.index.isin(negatives.Name)]
    negatives_degnorm = negatives[negatives.Name.isin(neg_w_degrees[neg_w_degrees.Degree >= min_degree].index)]
    if verbose:
        print ("{} genes have a degree >= {}.".format(negatives_degnorm.shape[0], min_degree))

    return negatives_degnorm


def write_hdf5_container(fname, adj, F, node_names, y_train, y_val, y_test, train_mask, val_mask, test_mask, feature_names, features_raw):
    f = h5py.File(fname, 'w')
    string_dt = h5py.special_dtype(vlen=str)
    f.create_dataset('network', data=adj, shape=adj.shape)
    f.create_dataset('features', data=F, shape=F.shape)
    f.create_dataset('gene_names', data=node_names, dtype=string_dt)
    f.create_dataset('y_train', data=y_train, shape=y_train.shape)
    f.create_dataset('y_val', data=y_val, shape=y_val.shape)
    f.create_dataset('y_test', data=y_test, shape=y_test.shape)
    f.create_dataset('mask_train', data=train_mask, shape=train_mask.shape)
    f.create_dataset('mask_val', data=val_mask, shape=val_mask.shape)
    f.create_dataset('mask_test', data=test_mask, shape=test_mask.shape)
    f.create_dataset('feature_names', data=np.array(feature_names, dtype=object), dtype=string_dt)
    f.create_dataset('features_raw', data=features_raw, shape=features_raw.shape)
    f.close()

    print ("Container written to {}".format(fname))