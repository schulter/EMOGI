#author: Roman Schulte-Sasse
# date: 20.06.2018

import pandas as pd
import numpy as np
import os, h5py, argparse, sys
from collections import Counter
from joblib import Parallel, delayed


def load_annotation_gff(annotation_path, gene_list=None, meth_data=None):
    """Load a GFF annotation file from disk into a pandas DF.

    This function will read a genome annotation file (GFF3 format) from disk
    into a dataframe. All annotations that do not originate from genes will
    be removed, together with NAs and non-relevant genes (optional, only when
    'gene_list' is given).

    Parameters:
    ----------
    annotation_path:            The path to the GFF3 annotation file

    gene_list:                  Dataframe with relevant genes as columns.
                                Specifies the list of genes to keep the
                                annotation for.
                                If None, all the annotations are returned.
    meth_data:                  Specify a methylation file to use for the
                                extraction of promoter regions for each of
                                the genes. If None, a simple method is used
                                that just takes the 500 bp region around the
                                TSS.

    Returns:
    -------
    A pandas DF containing the annotations. All NAs are deleted and only
    genes are kept (type column == 'gene'). Furthermore, promoter regions
    are already extracted and gene names and IDs are present as distinct
    columns.
    If a gene_list is given, only annotations for these genes are kept
    while all other genes are removed.
    """
    a_df = pd.read_csv(annotation_path, sep='\t', skiprows=7,
                       header=None,
                       names=['chr', 'source', 'type', 'start', 'end', 'score',
                              'strand', 'phase', 'attr']
                       )
    # drop all NAs
    a_df.dropna(axis=0, inplace=True)
    # remove everything that is not a gene
    a_df = a_df[a_df.type == 'gene']
    # put gene names and ENSEMBL IDs as extra columns
    annotated_gene_ids = [i[0].split('=')[1].split('.')[0].strip() for i in a_df.attr.str.split(';')]
    annotated_gene_names = [i[3].strip().split('=')[1].strip() for i in a_df.attr.str.split(';')]
    a_df['ID'] = annotated_gene_ids
    a_df['Symbol'] = annotated_gene_names

    # remove duplicate genes (not interested in transcript level)
    a_df.drop_duplicates(subset='Symbol', inplace=True)

    # extract promoter regions for the genes
    if meth_data is None: # use simple 500bp approach
        promoter_vec = np.vectorize(get_promotor_window)
        p_starts, p_ends = promoter_vec(a_df.start, a_df.end, a_df.strand)
    elif 'promoter_start' in a_df.columns and 'promoter_end' in a_df.columns:
        p_starts = a_df.promoter_start # don't do anything
        p_ends = a_df.promoter_start
    else: # use sliding window approach from Lisa
        x = a_df.apply(promoter_window_wrapper, axis=1, m_df=meth_data)
        p_starts = np.array([i[0] for i in x])
        p_ends = np.array([i[1] for i in x])
    a_df['promoter_start'] = p_starts
    a_df['promoter_end'] = p_ends

    # remove all genes that are not in the gene list
    if not gene_list is None:
        a_df = a_df[a_df.Symbol.isin(gene_list.Symbol)]
    return a_df


def load_methylation_file(path):
    m_df = pd.read_csv(path, sep='\t')
    m_df.dropna(axis=0, inplace=True)
    return m_df


def load_relevant_genes(container_path):
    with h5py.File(container_path, 'r') as f:
        node_names = f['gene_names'][:]
    # build data frame for node names and return
    return pd.DataFrame(node_names, columns=['ID', 'Symbol'])


def promoter_window_wrapper(row, m_df):
    return get_promotor_window(row['start'], row['end'], row['strand'], m_df)


def get_promotor_window(start, end, strand, meth_data=None):
    if strand == '+':
        tss = start
    else:
        tss = end
    if not meth_data is None: # fancy method to get promoter
        scan_region = (np.max(tss-1000, 0), tss + 1000)
        return calculate_promoter_window(scan_region[0], scan_region[1], meth_data)
    else:
        return np.max(tss-250, 0), tss + 250


def calculate_promoter_window(scan_start, scan_end, meth_data, shift=50, size=200):
    best_mean_meth = None
    best_window = (None, None)
    for i in range(int(scan_start), int(scan_end-size+1), shift):
        # get mean meth for window
        m_sites_window = meth_data[meth_data.Start.between(i, i+size)]
        m_mean_window = m_sites_window.Beta_value_mean.mean()
        
        # decide on best window
        if best_mean_meth is None: # first window
            best_mean_meth = m_mean_window
            best_window = i, i+size
        else: # any of the later windows
            if abs(best_mean_meth-m_mean_window) > 0.25: # large change in mean detected
                break
            else: # make the current window the best one
                best_mean_meth = m_mean_window
                best_window = i, i+size
    return best_window


def get_filenames(meth_raw_dir):
    """Extract all valid methylation files in dir.
    """
    all_filenames = []
    skipped_files = 0
    for dname in os.listdir(meth_raw_dir):
        sub_dirname = os.path.join(meth_raw_dir, dname)
        if os.path.isdir(sub_dirname):
            for fname in os.listdir(sub_dirname):
                if fname.endswith('gdc_hg38.txt'):
                    # don't use 27k methylation arrays
                    if fname.split('.')[2].endswith('450'):
                        all_filenames.append(os.path.join(sub_dirname, fname))
                    else:
                        skipped_files += 1
    print ("Skipped {} 27k arrays".format(skipped_files))
    return all_filenames


def get_mean_betaval_for_sample(annotation_genes, methylation_levels):
    """Extract mean methylation levels in promoters/gene bodies for each annotated gene.
    
    This function looks for all the methylation sites that fall
    inside all the promoters and gene bodies in the annotation genes df.
    The annotation_genes df is expected to contain the columns
    'promoter_start' and 'promoter_end'.
    
    Parameters:
    ----------
    annotation_genes:           A dataframe containing a gene per row.
                                It needs to have columns for promoter_start
                                and promoter_end.
    methylation_levels:         A dataframe that contains cpg sites per row.
                                The sites are expected to be one bp long and
                                should have the columns: 'Start', 'End' and 
                                Beta_value'.
    Returns:
    Two lists with the mean beta values for each promoter and the number of cpg sites
    that support the promoter. The first list contains the mean beta values and
    the second list contains the support.
    """
    beta_vals_prom = []
    beta_vals_gene = []
    n_supports_prom = []
    n_supports_gene = []
    count = 0
    for _, row in annotation_genes.iterrows():
        # restrict possible sites to same chr
        cpgs_on_chr = methylation_levels[methylation_levels.Chromosome == row.chr]
        # promoter
        m_sites_in_promoter = cpgs_on_chr[cpgs_on_chr.Start.between(row.promoter_start, row.promoter_end)]
        beta_vals_prom.append(m_sites_in_promoter.Beta_value.mean())
        n_supports_prom.append(m_sites_in_promoter.shape[0])
        # gene body. Don't start with the start and end coordinates but rather with the end of the promoter
        if row.strand == '+':
            m_sites_gene = cpgs_on_chr[cpgs_on_chr.Start.between(row.promoter_end, row.end)]
        else:
            m_sites_gene = cpgs_on_chr[cpgs_on_chr.Start.between(row.start, row.promoter_start)]
        beta_vals_gene.append(m_sites_gene.Beta_value.mean())
        n_supports_gene.append(m_sites_gene.shape[0])
        count += 1
    return beta_vals_prom, beta_vals_gene, n_supports_prom, n_supports_gene


def get_float(string):
    """ Convert a string into a float.

    This function converts a string to float if the string represents a floating
    point number (or integer). If that is not the case, it returns NaN.

    Parameters:
    ----------
    string:                     The string that is to be converted to float.

    Returns:
    A floating point number. Either with the value from `string` or numpy.nan
    """
    try:
        return float(string)
    except:
        return np.inf

def get_closest_gene(row):
    """ Get the closest gene from a row of TCGA level 3 methylation data.

    Extracts the gene name of the closest transcript promoter from
    a row of TCGA DNA methylation data (level 3). Each of the lines
    contain the coordinates of the CpG site together with information
    on the closest gene around it. There are three relevant fields (columns)
    for that: Gene_Symbol, Transcript_ID and Position_to_TSS.
    
    Parameters:
    ----------
    row:                        A row from a TCGA DNA methylation Dataframe.
                                Should contain the columns `Position_to_TSS`
                                and `Gene_Symbol`.
    """
    # extract rows
    genes = np.array(row.Gene_Symbol.split(';'))
    protein_coding_genes = np.array([i == 'protein_coding' for i in row.Gene_Type.split(';')])
    dists = np.array([get_float(i) for i in row.Position_to_TSS.split(';')])

    # remove non-protein-coding genes
    dists = dists[protein_coding_genes]
    genes = genes[protein_coding_genes]

    # return closest gene and distance
    if dists.shape[0] > 0:
        idx = np.argmin(np.abs(dists))
        return genes[idx], dists[idx]
    else:
        return None, None

def get_closest_5prime_transcript(row, gene_transcript_mapping):
    """ Get the closest 5' transcript from a row of TCGA level 3 methylation data.

    Extracts the transcript ID of the transcript whose TSS is closest to the
    CpG site in question. This is computed from a row of TCGA DNA methylation
    data (level 3). The function uses a pre-computed mapping from genes
    to 5' transcripts for its calculation.
    It first extracts the transcripts and distances to TSS for the CpG site
    and then discards all that are not in the list of closest transcripts in
    the mapping. Finally, from the valid transcripts the one with the closest
    TSS is chosen and returned.
    
    Parameters:
    ----------
    row:                        A row from a TCGA DNA methylation Dataframe.
                                Should contain the columns `Position_to_TSS`
                                and `Gene_Symbol`.
    
    Returns:
    The transcript ID and distance to the transcript TSS of a CpG site.
    This can be used as part of an apply call on a dataframe.
    """
    # extract rows
    transcripts = pd.Series(row.Transcript_ID.split(';'))
    transcripts = transcripts.str.split('.').str[0] # remove transcript version number
    
    valid_transcripts = transcripts.isin(gene_transcript_mapping)
    dists = np.array([get_float(i) for i in row.Position_to_TSS.split(';')])

    # remove non-protein-coding genes
    dists = dists[valid_transcripts]
    transcripts = transcripts[valid_transcripts]

    if dists.shape[0] > 0:
        idx = np.argmin(np.abs(dists))
        return np.array(transcripts)[idx], dists[idx]
    else:
        return None, None


def get_promoter_betaval_tcgaannotation(methylation_levels):
    # write to DF which is the closest gene
    x = methylation_levels.apply(get_closest_gene, axis=1)
    methylation_levels['closest_gene'] = [i[0] for i in x]
    methylation_levels['dist_closest_gene'] = [i[1] for i in x]

    # filter
    close_cpgs = methylation_levels[methylation_levels.dist_closest_gene.abs() < 1000]

    # group and return
    cpgs_for_genes = close_cpgs.groupby('closest_gene')
    beta_values_prom = cpgs_for_genes.Beta_value.mean()
    n_supports_prom = cpgs_for_genes.Beta_value.count()

    return beta_values_prom, n_supports_prom


def get_gene2transcript_map(annotation_df):
    """Calculate a mapping from a gene to its most 5' transcript.

    This function calculates a mapping from a gene to its most 5' transcript using a
    GENCODE annotation file. For each transcript, the corresponding parent gene is
    extracted together with its distance to the chromosome start. For genes on the
    - strand, negative values are used.
    Using a map/reduce pattern, the transcripts are grouped by parent genes and the
    most 5' transcript per gene is chosen.

    Parameters:
    ----------
    annotation_df:              The GENCODE annotation file as pandas DataFrame.
                                Usually, the basic annotation is preferred over the
                                comprehensive.

    Returns:
    A DataFrame containing the gene symbol as index and the corresponding 5' transcript
    as column.
    """
    # get a mapping from gene to transcript
    transcripts = annotation_df[annotation_df.type == 'transcript'].copy()
    # extract all attributes of transcripts
    transcript_attrs = []
    for attrs in transcripts.attr.str.split(';'):
        transcript_attrs.append({i.split('=')[0]:i.split('=')[1] for i in attrs})

    # set the attributes to additional columns (remove transcript version number)
    transcripts['ID'] = [i['ID'].strip().split('.')[0] for i in transcript_attrs]
    transcripts['Parent'] = [i['Parent'] for i in transcript_attrs]
    transcripts['Gene_Symbol'] = [i['gene_name'] for i in transcript_attrs]

    # build column with the distance to the chromosome start (for transcripts on the - strand, take negative values)
    transcripts['dist_to_chr_start'] = transcripts.start
    transcripts.loc[transcripts.strand == '-', 'dist_to_chr_start'] = -transcripts.end
    transcripts.set_index('ID', inplace=True)
    # mapping is now simply the transcript in a group of a gene that has minimum distance
    gene_transcript_mapping = transcripts.groupby('Gene_Symbol').dist_to_chr_start.idxmin()

    return gene_transcript_mapping


def get_cpg_transcript_map(annotation_df, methylation_levels):
    """Calculates a mapping from CpG site to closest transcript.
    
    This function calculates a mapping from CpG site to gene promoter and
    the distance to it. It makes use of the annotations in a TCGA methylation
    file (level 3). For each CpG site, its closest transcripts and their distance
    to the TSS are extracted. Then, all transcripts that are not the most 5'
    transcripts of a gene are discarded and from the remaining transcripts, the
    closest one is chosen. This implies that a CpG site can only belong to one
    gene promoter.
    The calculated map of CpG sites to transcript promoters is finally joined
    with a mapping from transcripts to genes (computed from GENCODE annotation)
    and the final mapping is returned.

    Parameters:
    ----------
    annotation_df:              The GENCODE annotation file as pandas DataFrame.
                                Usually, the basic annotation is preferred over the
                                comprehensive.
    methylation_levels:         A TCGA level 3 methylation file. This will
                                serve as template for all other samples, so
                                if you use 450k data, this file should contain
                                all the CpG sites you are interested in.

    Returns:
    A Dataframe with CpG sites as rows (Composite Element REF) and the gene/transcript
    as columns. The columns contain the distance to the closest transcript of that CpG
    site, the Ensembl ID of that transcript and the gene name of the transcript (Symbol).
    """
    gene_transcript_map = get_gene2transcript_map(annotation_df)
    # get the closest 5' transcript according to the gene transcript map
    x = methylation_levels.apply(get_closest_5prime_transcript,
                                 axis=1,
                                 gene_transcript_mapping=gene_transcript_map)
    methylation_levels['closest_transcript'] = [i[0] for i in x]
    methylation_levels['dist_closest_transcript'] = [i[1] for i in x]
    
    # get transcript ID and Gene Symbol in mapping as new columns
    gene_transcript_map_df = pd.DataFrame(gene_transcript_map)
    gene_transcript_map_df.columns = ['Transcript_ID']
    gene_transcript_map_df['Symbol'] = gene_transcript_map_df.index

    # construct the mapping
    cpgs_with_genes = methylation_levels[~methylation_levels.closest_transcript.isnull()]
    cpgs_with_genes.drop(['Transcript_ID', 'Position_to_TSS'], axis=1, inplace=True)
    cpg_gene_mapping = cpgs_with_genes.merge(gene_transcript_map_df,
                                             left_on='closest_transcript',
                                             right_on='Transcript_ID'
                                            )
    cpg_gene_mapping.set_index('Composite Element REF', inplace=True)
    mapping_cols = ['dist_closest_transcript', 'Transcript_ID', 'Symbol']
    return cpg_gene_mapping[mapping_cols]


def get_meth_df_from_mapping(cpg_gene_map, path_name, clean=False):
    """Computes promoter DNA methylation from a CpG to gene mapping.

    This function computes the average DNA methylation at promoters from
    a mapping of CpG sites to genes. It essentially only joins the mapping
    with the actual beta values from the sample in question.
    It then aggregates the genes together, returning a DataFrame with
    the average DNA methylation at the promoter of each gene and the
    cancer type of the individual.

    Parameters:
    ----------
    cpg_gene_map:               A mapping between CpG sites and genes/transcripts.
                                Can be computed from TCGA annotations to the methylation
                                data. @see get_cpg_transcript_map
    path_name:                  The path to the methylation file. Should be in TCGA format,
                                thus containing information on close-by genes additionally
                                to the pure coordinates and beta value.
    clean:                      Whether to use potential previous runs and data written into
                                the download directory or not. If changing the method of how
                                DNA methylation is computed, clean=True is recommended.

    Returns:
    A dataframe with methylation levels at promoters and number of CpG sites per gene promoter as
    columns and gene names as rows and the cancer type of the sample.
    Ideally, there should be as many genes as there are in the mapping file.
    The columns of the returned dataframe will have the format:
    <sample_id>|<cancer_type>|<beta_val/support> (separator is |)
    """
    sub_dirname = os.path.dirname(path_name)
    fname = os.path.basename(path_name)
    avg_meth_result_path = os.path.join(sub_dirname,
            'avg_methylation_tcgaannot.tsv'
            )
    cancer_type = fname.split('.')[1].split('_')[1].strip().lower()
    sample_id = fname.split('.')[-3].strip()

    if not os.path.isfile(avg_meth_result_path) or clean:
        meth_df = load_methylation_file(path_name)
        meth_df.drop(['Transcript_ID', 'Gene_Symbol', 'Gene_Type', 'Position_to_TSS',
                      'CGI_Coordinate', 'Feature_Type'],
                      axis=1, inplace=True)
        meth_df.set_index('Composite Element REF', inplace=True)
        meth_annot = meth_df.join(cpg_gene_map)
        meth_annot_inrange = meth_annot[meth_annot.dist_closest_transcript.abs() < 1000]
        res = pd.DataFrame(meth_annot_inrange.dropna().groupby('Symbol').Beta_value.mean())
        res['support_promoter'] = meth_annot_inrange.dropna().groupby('Symbol').Beta_value.count()
        colnames = ['mean_beta_value_promoter', 'support_promoter']
        cols = ['{}|{}|{}'.format(sample_id, cancer_type, i) for i in colnames]
        res.columns = cols
        res.to_csv(avg_meth_result_path, sep='\t')
    else:
        res = pd.read_csv(avg_meth_result_path, sep='\t')
        res.set_index('Symbol', inplace=True)
    return res, cancer_type


def get_meth_df_for_sample(annotation_df, path_name, clean=False, tcga_annot=True):
    """Calculates the mean methylation DF for one sample.
    
    This function computes the mean methylation level at
    the promoter and gene body for all genes in the annotation
    dataframe using the methylation data specified in the path
    name. Note that this must be in the TCGA methylation format.
    The annotation dataframe is expected to contain the promoter
    region in the columns 'promoter_start' and 'promoter_end'.
    If the directory of the file given by 'path_name' contains
    already a file 'avg_methylation.tsv', this will be loaded
    and returned simply instead of re-extracting the methylation
    values. If clean is set to True, the extraction takes place
    in any case.

    Parameters:
    ----------
    annotation_df:              The annotation file as dataframe. Can be computed using
                                @see load_annotation_gff from the gencode annotation.
    path_name:                  The path to the methylation file. Should be in TCGA format,
                                thus containing information on close-by genes additionally
                                to the pure coordinates and beta value.
    clean:                      Whether to use potential previous runs and data written into
                                the download directory or not. If changing the method of how
                                DNA methylation is computed, clean=True is recommended.
    tcga_annot:                 Whether or not to use the annotation from TCGA to each
                                individual CpG site. Otherwise it uses promoter windows
                                around the TSS of a gene.

    Returns:
    A dataframe with methylation levels at promoters and number of CpG sites per gene promoter as
    columns and gene names as rows and the cancer type of the sample.
    Ideally, there should be as many genes as there are in the mapping file.
    The columns of the returned dataframe will have the format:
    <sample_id>|<cancer_type>|<beta_val/support> (separator is |)
    """
    sub_dirname = os.path.dirname(path_name)
    fname = os.path.basename(path_name)
    if tcga_annot:
        avg_meth_result_path = os.path.join(sub_dirname,
                'avg_methylation_tcgaannot.tsv'
                )
    else:
        avg_meth_result_path = os.path.join(sub_dirname,
                        'avg_methylation.tsv'
                        )
    # get the cancer type first
    cancer_type = fname.split('.')[1].split('_')[1].strip().lower()
    sample_id = fname.split('.')[-3].strip()
    if not os.path.isfile(avg_meth_result_path) or clean:
        # get mean methylation levels around promoters
        meth_df = load_methylation_file(path_name)
        if tcga_annot:
            beta_prom, sup_prom = get_promoter_betaval_tcgaannotation(meth_df)
            res = pd.DataFrame([beta_prom, sup_prom]).T
            colnames = ['mean_beta_value_promoter', 'support_promoter']
            cols = ['{}|{}|{}'.format(sample_id, cancer_type, i) for i in colnames]
            res.columns = cols
        else:
            beta_p, beta_g, sup_p, sup_g = get_mean_betaval_for_sample(annotation_df,
                                                                       meth_df)
            # construct dataframe from results
            res = pd.DataFrame([beta_p, sup_p, beta_g, sup_g]).T
            colnames = ['mean_beta_value_promoter', 'support_promoter',
                        'mean_beta_value_gene', 'support_gene']
            cols = ['{}|{}|{}'.format(sample_id, cancer_type, i) for i in colnames]
            res.columns = cols
            res.set_index(annotation_df.Symbol, inplace=True) # assumes same order

        # write the average beta values (at promoters) to file
        res.to_csv(avg_meth_result_path, sep='\t')
    else:
        res = pd.read_csv(avg_meth_result_path, sep='\t')
        res.set_index('Symbol', inplace=True)
    return res, cancer_type


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Preprocess methylation data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--annotation',
                        help='Path to annotation file (GFF3)',
                        dest='annotation',
                        type=str
                        )
    parser.add_argument('-md', '--methylation_dir',
                        help='Path to methylation download dir',
                        dest='meth_dir',
                        type=str,
                        required=True
                        )
    parser.add_argument('-rg', '--rel_genes',
                        help='Path to container with relevant genes',
                        dest='rel_genes',
                        default=None,
                        type=str
                        )
    parser.add_argument('-c', '--clean',
                        help='Re-compute when there are already avg files?',
                        dest='clean',
                        default=False,
                        type=bool
                        )
    parser.add_argument('-tcga', '--tcga_annotations',
                        help='Use TCGA annotations to CpG sites?',
                        dest='tcga_annot',
                        default=True,
                        type=bool
                        )
    parser.add_argument('-fiveprime', '--tcga_most_downstream_transcript',
                        help='Only use DNA methylation of most 5prime transcript?',
                        dest='fiveprime',
                        default=True,
                        type=bool
                        )
    parser.add_argument('-o', '--output',
                        help='Path to output Dataframe (huge sample-wise matrix)',
                        dest='output',
                        type=str,
                        required=True
                        )
    parser.add_argument('-n', '--cores',
                        help='Number of cores to use',
                        dest='n_jobs',
                        default=1,
                        type=int
                        )
    args = parser.parse_args()

    # load data and compute promoter windows if required
    if not args.rel_genes is None:
        relevant_genes = load_relevant_genes(args.rel_genes)
    else:
        relevant_genes = None
    if args.annotation.endswith('.tsv'):
        annotation_df = pd.read_csv(args.annotation, sep='\t')
    elif args.annotation.endswith('.gff3'):
        annotation_df = load_annotation_gff(args.annotation, gene_list=relevant_genes)
    else:
        print ("Unknown Annotation file format. Only tsv and gff3 are supported")
        sys.exit(-1)
    print ("Loaded Annotation file with {} genes".format(annotation_df.shape[0]))

    # get file names of DNA methylation downloads
    all_files = get_filenames(args.meth_dir)
    print ("Found {} methylation profiles".format(len(all_files)))

    # Derive DNA methylation at promoters
    if args.tcga_annot and args.fiveprime: # use TCGA annotations and only fiveprime transcript
        # get mapping of CpG site to closest 5' transcript
        if args.annotation.endswith('.gff3'):
            a_df = pd.read_csv(args.annotation, sep='\t', skiprows=7,
                            header=None,
                            names=['chr', 'source', 'type', 'start', 'end', 'score',
                                    'strand', 'phase', 'attr']
                            )
        else:
            a_df = pd.read_csv(args.annotation, sep='\t')
        cpg_genemap_file = os.path.join(os.path.dirname(args.annotation), 'cpg2genemap.tsv')
        if not os.path.exists(cpg_genemap_file):
            print ("Computing CpG site to transcript map")
            cpg_mapping = get_cpg_transcript_map(a_df, load_methylation_file(all_files[0]))
            cpg_mapping.to_csv(cpg_genemap_file, sep='\t')
            print ("Mapping computed. Applying for all samples now...")
        else:
            print ("Loading mapping of CpG site to transcripts from disk...")
            cpg_mapping = pd.read_csv(cpg_genemap_file, sep='\t')
            cpg_mapping.set_index('Composite Element REF', inplace=True)
        
        # use the mapping as template for all samples (e.g. apply)
        print ("Applying cpg 2 transcript mapping for all samples (takes time and uses processors)")
        results = Parallel(n_jobs=args.n_jobs)(delayed(get_meth_df_from_mapping)(cpg_mapping,
                                                                                 f,
                                                                                 args.clean) for f in all_files)
    else: # use TCGA annotations (and mean over all transcripts of each gene) or fixed promoters
        results = Parallel(n_jobs=args.n_jobs)(delayed(get_meth_df_for_sample)(annotation_df,
                                                                               f,
                                                                               args.clean,
                                                                               args.tcga_annot) for f in all_files)
    all_samples_preprocessed = [i[0] for i in results]
    cancer_types = [i[1] for i in results]

    # some information on how many samples we have
    for key, value in Counter(cancer_types).items():
        print ("Found {} samples for Cancer {}".format(value, key))

    # join samples to form one df with all samples as columns
    total_df = pd.concat(all_samples_preprocessed, axis=1)
    total_df.to_csv(args.output, sep='\t', compression='gzip')

    print ("Finished. Written mean methylation matrix for {} samples to disk".format(len(all_samples_preprocessed)))
