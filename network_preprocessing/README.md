# Network Preprocessing
EMOGI was tested with 5 different protein-protein interaction (PPI )networks, namely:
* [IRefIndex](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-405)
* [ConsensusPathDB](http://cpdb.molgen.mpg.de/)
* [PCNet](https://www.cell.com/cell-systems/pdf/S2405-4712(18)30095-4.pdf)
* [Multinet](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002886)
* [STRING-db](https://string-db.org/)

While PCNet and Multinet were not further preprocessed, ID mapping and thresholding edges was done for the other networks.
The notebooks in this folder can be used to process each of the PPI networks in order to obtain a valid network to work with.
The files produced from the notebooks can be read by common methods from networkx or pandas.
