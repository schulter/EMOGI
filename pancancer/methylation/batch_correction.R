library(sva)

args <- commandArgs(trailingOnly = TRUE)

path_to_samples = args[1]
phenotype_path = args[2]
output_path = args[3]

# read samples and phenotype matrices
samples = read.table(path_to_samples, sep = '\t', header = TRUE, row.names = 1)
pheno = read.table(phenotype_path, sep = '\t', header = TRUE)
sample_matrix = as.matrix(samples)
pheno$batch <- as.factor(pheno$batch)

freqs_of_batches = as.data.frame(table(pheno$batch))

# run ComBat using the batch variable for adjustment
if (length(unique(pheno$batch)) > 1 && max(freqs_of_batches$Freq) > 1) { # something to adjust
    modCombat = model.matrix(~1, data = pheno)
    adjusted_data = ComBat(dat = sample_matrix, batch = as.factor(pheno$batch), mod = modCombat, par.prior = TRUE, prior.plots = FALSE)
    write.table(adjusted_data, output_path, sep = '\t')
} else { # everything comes from the same batch, nothing to do
    print("Don't normalize because same batch!")
  write.table(sample_matrix, output_path, sep = '\t')
}