suppressMessages(library('GOstats'))
suppressMessages(library("org.Hs.eg.db"))
suppressMessages(library("AnnotationDbi"))
suppressMessages(library("annotate"))
suppressMessages(library("GO.db"))
suppressMessages(library("KEGG.db"))
suppressMessages(library("biomaRt"))
suppressMessages(library(gplots))
suppressMessages(library("GSEABase"))
suppressMessages(library(pathview))

# specify via command line path of pagerank scores, number of genes & output path
args <- commandArgs(TRUE)
input_path <- args[1]
output_path <- args[2]
number_of_genes <- as.integer(args[3])

# define Biomart (which helps getting entrez gene IDs from ensembl gene IDs)
ensembl = useMart("ensembl")
ensembl = useDataset("hsapiens_gene_ensembl", mart=ensembl)
#listAttributes(ensembl)

# Define the Universe
#universe <- read.table("/home/caffrey/bin/entrez-gene-names.txt", sep="\t", header=TRUE)  #CHANGE!!!!!!
#entrez_ids=universe[,1]
#entrezgene_ids_universe = getBM(attributes=c("entrezgene"), filters=c("ensembl_gene_id"), values=ensembl_ids, mart=ensembl)
#entrezgene_ids_universe=unique(entrezgene_ids_universe[!is.na(entrezgene_ids_universe)])
entrezgene_ids_universe <- NULL; # All entrez gene IDs associated with any gene ontology are used

full <- head(read.table(input_path, sep="\t", header=T), number_of_genes);
ensembl_ids <- full[1];

# get entrez gene ids from ensembl gene IDs
ensembl_entrez_ids <- getBM(values = ensembl_ids,
                            filters = "ensembl_gene_id",
                            mart = ensembl,
                            attributes = c("ensembl_gene_id", "entrezgene","hgnc_symbol")
                            )

#KEGG enrichment analysis
frame = toTable(org.Hs.egPATH)
keggframeData = data.frame(frame$path_id, frame$gene_id)
keggFrame=KEGGFrame(keggframeData,organism="Homo sapiens")
gsc <- GeneSetCollection(keggFrame, setType = KEGGCollection())

# perform the actual test
params <- GSEAKEGGHyperGParams(name="GSEA - PageRank Enrichment",
                               geneSetCollection=gsc,
                               geneIds = ensembl_entrez_ids$entrezgene,
                               universeGeneIds =  entrezgene_ids_universe,
                               pvalueCutoff = 0.1,
                               testDirection = "over"
)

# write out results
test_results <- hyperGTest(params)
results_summary <- data.frame(summary(test_results))
write.table(results_summary, output_path, sep="\t")
print("Successfully written results from Pathway analysis")