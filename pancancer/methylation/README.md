# Preprocessing DNA Methylation Data
The scripts and code in this directory let you preprocess DNA methylation data from 450k Illumina bead arrays.
The [notebook](build_methylation_container.ipynb) can show you how to combine tumor and normal DNA methylation data
while the [script](get_mean_sample_meth.py) lets you compute the average promoter and gene body methylation
given some genome annotation.

## Compute Mean Promoter Methylation
There are multiple options to compute the promoter DNA methylation and the script allows for multiple ones. We decided to
define a promoter based on the most 5`transcript for that gene. We then define the promoter as the region +-1000 base pairs
around the TSS. The script works by first computing a map where each measured CpG site is assigned to a gene together with the
distance to the TSS. We then use that mapping and apply it to all samples to reduce runtime.
You can run the script using:
```
python get_mean_sample_meth.py --annotation <path-to-annotation-gff3 file> --methylation-dir <path-to-downloaded-TCGA-methylation-data> --output <path-to-gene-sample-matrix>
```

## Batch Correction
We provided a [notebook](batchcorrection.ipynb) and an R script to do batch correction using
[ComBat](https://bioconductor.org/packages/release/bioc/vignettes/sva/inst/doc/sva.pdf). No matter what you use, you have to
make sure that all packages are installed for the R script because the notebook only calls the R script multiple times.
We use the plate numbers as batch variables to normalize against and normalize each study individually.

## Computing Differential DNA Methylation
One more [notebook](build_methylation_container.ipynb) allows you to compute differential DNA methylation values (subtracted
normal and tumor beta values) for promoters and gene bodies of genes.
It provides smoe plots, including a UMAP of the samples colored by cancer type of the differential values.

The notebook writes a gene-sample matrix to HDF5 and a gene-cancer type matrix to a tab-separated value file.
