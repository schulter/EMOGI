library(DESeq2)
ge <- read.table('data/differential_expression/readcounts_raw.tsv', row.names = 3, header = TRUE,sep = '\t')
todrop <- c("Lp.T0.11_S2_R1.bam", "Lp.T0.1_S1_R1.bam", "Lp.T0.21_S3_R1.bam", "Unnamed..0")
ge <- ge[, !(names(ge) %in% todrop)]
colnames(ge) <- gsub("\\.bam", "", colnames(ge))
colnames(ge) <- gsub("\\.", "_", colnames(ge))
coldata <- read.table('data/differential_expression/coldata.tsv', header = TRUE, sep = '\t', row.names=1)
all(rownames(coldata) == colnames(ge))

dds <- DESeqDataSetFromMatrix(countData = ge, colData = coldata, design = ~ condition)

dds <- DESeq(dds)
res <- results(dds, name="condition_gfppT16_vs_ControlT16")
write.csv(as.data.frame(res), file = "data/differential_expression/deseq2_gfppT16_vs_ControlT16.csv")
