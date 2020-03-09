# Preprocessing of Gene Expression Data where 
We used preprocessed gene expression data from [Wang et al.](https://www.nature.com/articles/sdata201861) where the authors
used GTEx data to normalize TCGA tumor mRNA-seq data. We used the batch corrected and normalized data as recommended by the
authors and computed *log_2 fold changes* for all genes. The [notebook](process_geneexpression_datadescriptor.ipynb) can be
parametrized to either use GTEx or TCGA normal tissue gene expression for normalization.
We did a rather throrough analysis and found both data normalization strategies to be suboptimal. For some genes, we fail
completely to reproduce expected behavior while for others, the results seem to make perfect sense.
The notebook provides MA plots, gene-wise (and general) expression plots over cancer types, histograms and a UMAP embedding
of samples colored by the cancer type. We decided to use GTEx normalization of the normalized and batch corrected data from
Wang et al. because that yielded the UMAP embedding where tumor and normal tissues were closest.

The script writes a gene x sample matrix to a HDF5 container and a gene x cancer type matrix with the average fold changes of that gene in the cancer type as tab-separated value file.
