# Preprocessing Mutation Frequencies
We provide notebooks to preprocess SNV and CNA information from the TCGA.

## Preprocessing CNAs
We collected copy number information from GISTIC 2.0 via firebrowse. The preprocessing extracts targets of copy number
aberrations (both, amplified and deleted regions). The data has to be downloaded, this can be done with the
[firehose_get](https://confluence.broadinstitute.org/display/GDAC/Download) tools using:
```
./firehose_get -tasks gistic analysis latest
```
The [notebook](mutfreq/preprocess_cnas.ipynb) writes two matrices to file,
containing the average copy number effect frequency (mean over samples for a cancer type) per cancer type and gene as well as
a matrix with the copy number changes per sample and gene. It further plots distributions over cancer types and computes a UMAP
embedding of the samples, colored by cancer type.

## Preprocessing of SNVs
The [notebook](mutfreq/preprocess_mutation_freqs.ipynb) preprocesses single nucleotide variants from MAF files and then
(optionally) uses the already computed CNA frequencies together with SNVs to copmute mutation frequencies per gene and sample.
The notebook can normalize SNV frequencies for exonic gene length if GENCODE annotation is provided.
Again, the script offers ways to compute UMAP embeddings of samples colored by cancer type and basic plots that verify the
correct preprocessing steps.

The script writes a HDF5 container with gene x sample matrix, CNA sample matrix and SNV sample matrix to disk. It further
writes also the mean mutation frequencies per cancer type (gene x cancer type matrix) to a tab-separated file.
