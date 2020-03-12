# Preprocessing of TCGA Data
This folder contains scripts to process and normalize:
* [Mutation MAF files](mutfreq/README.md) (can be downloaded via TCGA data portal),
* [CNA copy number information](mutfreq/README.md) as computed by GISTIC 2.0 (can be downloaded via [firehose](https://gdac.broadinstitute.org/))
* [DNA methylation data](methylation/README.md) from Illumina 450k bead arrays (can be downloaded again from TCGA data portal)
* [Gene expression data](expression/README.md) (normalized data available in the publication from "Data Descriptor: Unifying cancer and normal RNA sequencing data from different sources" Wang et al., 2018).

The [build_multiomics_container](preprocessing/build_multiomics_container.ipynb) notebook then takes all the individually preprocessed data along with a PPI network of choice and constructs the HDF5 container.

Each of the individual omics data types has its own readme file and many individual parameters for normalization and preprocessing.
Also the PPI network preprocessing is explained in more detail in the [respective readme file](../network_preprocessing/README.md).
