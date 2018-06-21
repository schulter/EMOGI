import os.path
import argparse
import numpy

# ===============================================================
# read terminal parameters
# ===============================================================


def args():
    args_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args_parser.add_argument(
        "-a", "--gffFile", help="gff file with promoter/miRNA annotation", type=str, required=True)
    args_parser.add_argument(
        "-b", "--bedFile", help="bed file with annotation of DNA methylation regions", type=str, required=True)
    args_parser.add_argument(
        "-v", "--betaValuesFile", help="file with beta values and ID of DNA methylation", type=str, required=True)
    args_parser.add_argument(
        "-n", "--minNumSites", help="minimum number of DNA methylation sites per region; Default: 1 site", type=str, default=1)
    args_parser.add_argument(
        "-o", "--outputDir", help="output directory for bed analysis files (not required): default: ./", type=str, default="./")
    return args_parser.parse_args()

# ===============================================================
# Methods
# ===============================================================


def proofFile(filename):
    if not (os.path.exists(filename)):
        exit("{0} does not exist!".format(filename))
        return False
    if not (os.path.isfile(filename)):
        exit("{0} is not a file!".format(filename))
        return False
    return True

# ===============================================================


def setOutputFilenames(bedFile, gffFile, outputDir):
    bed = bedFile.split("/")
    bed = bed[len(bed)-1].split(".")
    bed = "_".join(bed[0:(len(bed)-1)])

    gff = gffFile.split("/")
    gff = gff[len(gff)-1].split(".")
    gff = "_".join(gff[0:(len(gff)-1)])

    global bedAnalysisFileName
    bedAnalysisFileName = bed + "-" + gff
    global bedAnalysisFile
    global DNAmethFile
    bedAnalysisFile = outputDir + bedAnalysisFileName + ".intersect"
    DNAmethFile = outputDir + bedAnalysisFileName + ".gff"

# ===============================================================


def createMappingBetaValues(betaValuesFile):
    with open(betaValuesFile) as readhandle:
        mapping_betaValues = {}
        for line in readhandle:
            line = line.rstrip().split('\t')
            betaValue = float(line[1])
            mapping_betaValues[str(line[0])] = betaValue
    return mapping_betaValues

# ===============================================================


def getShiftedWindows(chrom, start, end, strand, shift, window_size, optimal_window_file):
    with open(optimal_window_file, 'w') as writehandle:
        for i in range(int(start), int((end - window_size + 1)), shift):
            line = chrom + '\t' + str(int(i)) + '\t' + str(int(i+window_size)) + \
                '\t' + '.' + '\t' + '.' + '\t' + strand + '\n'
            writehandle.write(line)

# ===============================================================


def calculateOptimalWindow(chrom, middle, strand, bedFile, mapping_betaValues, region):

    shift = 50  # bp
    window_size = 200  # bp

    # upstream
    if region == 'u':
        if strand == '+':
            scanningRegion_start = max(middle - 1000, 0)
            scanningRegion_end = middle
        else:
            scanningRegion_start = middle
            scanningRegion_end = middle + 1000
    # downstream
    if region == 'd':
        if strand == '+':
            scanningRegion_start = middle
            scanningRegion_end = middle + 1000
        else:
            scanningRegion_start = max(middle - 1000, 0)
            scanningRegion_end = middle

    optimal_window_file = outputDir + 'tmp_optimal_window.bed'
    bed_analysis_results = outputDir + 'tmp_coverage.bed'
    getShiftedWindows(chrom, scanningRegion_start, scanningRegion_end,
                      strand, shift, window_size, optimal_window_file)

    bedCmd = "intersectBed -wo -s -a " + optimal_window_file + \
        " -b " + bedFile + " > " + bed_analysis_results
    os.system(bedCmd)

    readhandle = open(bed_analysis_results)
    IDtoDNAmeth = {}
    for line in readhandle:
        line = line.rstrip().split('\t')
        DNAmeth_ID = line[9]
        TSS_ID = (line[1], line[2])
        if str(DNAmeth_ID) in mapping_betaValues:
            betaValue = mapping_betaValues[str(DNAmeth_ID)]
        if TSS_ID in IDtoDNAmeth:
            IDtoDNAmeth[TSS_ID].append(betaValue)
        if not TSS_ID in IDtoDNAmeth:
            IDtoDNAmeth[TSS_ID] = [betaValue]

    finalInterval = []
    betaValue_list = []

    if (region == 'u' and strand == '+') or (region == 'd' and strand == '-'):
        for key in sorted(IDtoDNAmeth.keys(), reverse=True):
            start = key[0]
            end = key[1]
            if finalInterval == []:
                finalInterval = [int(start), int(end)]
                betaValue_list.extend(IDtoDNAmeth[key])
            else:
                if finalInterval[0] < int(end) and abs(numpy.mean(betaValue_list) - numpy.mean(IDtoDNAmeth[key])) < 0.3:
                    finalInterval[0] = int(start)
                    betaValue_list.extend(IDtoDNAmeth[key])
                else:
                    break

    if (region == 'd' and strand == '+') or (region == 'u' and strand == '-'):
        for key in sorted(IDtoDNAmeth.keys()):
            start = key[0]
            end = key[1]
            if finalInterval == []:
                finalInterval = [int(start), int(end)]
                betaValue_list.extend(IDtoDNAmeth[key])
            else:
                if finalInterval[1] > int(start) and abs(numpy.mean(betaValue_list) - numpy.mean(IDtoDNAmeth[key])) < 0.3:
                    finalInterval[1] = int(end)
                    betaValue_list.extend(IDtoDNAmeth[key])
                else:
                    break

    os.system('rm ' + optimal_window_file)
    os.system('rm ' + bed_analysis_results)
    return finalInterval

# ===============================================================


def getOptimalWindow(chrom, start, end, strand, bedFile, mapping_betaValues):
    middle = int(start) + (int(end)-int(start))/2

    upstream_interval = calculateOptimalWindow(
        chrom, middle, strand, bedFile, mapping_betaValues, 'u')
    downstream_interval = calculateOptimalWindow(
        chrom, middle, strand, bedFile, mapping_betaValues, 'd')

    if upstream_interval == []:
        if strand == '+':
            upstream_interval = [middle-500, middle]
        else:
            upstream_interval = [middle, middle+500]

    if downstream_interval == []:
        if strand == '+':
            downstream_interval = [middle, middle+500]
        else:
            downstream_interval = [middle-500, middle]

    final_interval_start = min(upstream_interval + downstream_interval)
    final_interval_end = max(upstream_interval + downstream_interval)

    return [str(int(max(final_interval_start, 0))), str(int(final_interval_end))]

# ===============================================================


def getConsensusRegion(gffFile, bedFile, mapping_betaValues):
    bed_region = outputDir + 'new_promoter_regions.bed'
    #bed_results = './DNAmeth_coverage.bed'

    readhandle = open(gffFile)
    i = 0

    with open(bed_region, 'w') as writehandle:
        for line in readhandle:
            i = i+1
            print(i)
            line = line.rstrip().split("\t")
            promID = line[3]

            promoter_newRegion = getOptimalWindow(
                line[0], line[3], line[4], line[6], bedFile, mapping_betaValues)

            line[3] = promoter_newRegion[0]  # new start
            line[4] = promoter_newRegion[1]  # new end
            line[5] = promID  # save original start to know which promoter it was

            writehandle.write("\t".join(line) + "\n")

    bedCmd = "intersectBed -wo -a " + bed_region + \
        " -b " + bedFile + " > " + bedAnalysisFile
    os.system(bedCmd)

    readhandle.close()
    readhandle = open(bedAnalysisFile)
    IDtoDNAmeth = {}

    for line in readhandle:
        line = line.rstrip().split('\t')

        DNAmeth_ID = line[12]
        TSS_ID = line[5] + line[8]

        if str(DNAmeth_ID) in mapping_betaValues:
            betaValue = mapping_betaValues[str(DNAmeth_ID)]

            if TSS_ID in IDtoDNAmeth:
                IDtoDNAmeth[TSS_ID].append(betaValue)
            if not TSS_ID in IDtoDNAmeth:
                IDtoDNAmeth[TSS_ID] = [betaValue]

    readhandle.close()
    readhandle = open(gffFile)

    DNAmeth_coverage = []
    CpG_prom = []

    with open(DNAmethFile, 'w') as writehandle:
        for line in readhandle:
            line = line.rstrip().split('\t')
            promID = line[3] + line[8]

            if promID in IDtoDNAmeth:
                DNAmeth_coverage = float(
                    round(numpy.mean(IDtoDNAmeth[promID]), 3))
                CpG_prom = 'CpG'

                beta_values = IDtoDNAmeth[promID]
                unmeth = 0
                meth = 0
                # print(beta_values)
                for beta_value in beta_values:
                    if beta_value <= 0.2:
                        unmeth = unmeth + 1
                    if beta_value >= 0.6:
                        meth = meth + 1

                #writehandle.write('\t'.join(line) + '\t' + str(DNAmeth_coverage) + '\t' + CpG_prom + '\n')
                writehandle.write('\t'.join(line) + '\t' + str(DNAmeth_coverage) + '\t' + CpG_prom +
                                  '\t' + str(meth) + '\t' + str(unmeth) + '\t' + str(len(beta_values)) + '\n')

            else:
                DNAmeth_coverage = float(0)
                CpG_prom = 'unknown'

                #writehandle.write('\t'.join(line) + '\t' + str(DNAmeth_coverage) + '\t' + CpG_prom + '\n')
                writehandle.write('\t'.join(
                    line) + '\t' + str(DNAmeth_coverage) + '\t' + CpG_prom + '\t0\t0\t0\n')

    rmCmd = 'rm -f ' + bed_region
    os.system(rmCmd)

# ===============================================================


def plotDistribution(outputDir):
    rCmd = "R CMD BATCH --slave \"--args " + outputDir + " " + \
        bedAnalysisFileName + "\"  plotDistribution.R reportPlotDistribution.txt"
    os.system(rCmd)
    print("Plotting done!")

# ===============================================================
# Parameters
# ===============================================================

if __name__ == '__main__':
    parameters = args()

    bedFile = parameters.bedFile
    proofFile(bedFile)

    gffFile = parameters.gffFile
    proofFile(gffFile)

    betaValuesFile = parameters.betaValuesFile
    proofFile(betaValuesFile)

    minNumSites = parameters.minNumSites

    outputDir = parameters.outputDir

    # ===============================================================
    # executive part
    # ===============================================================

    if not outputDir.endswith("/"):
        outputDir = outputDir + "/"

    setOutputFilenames(bedFile, gffFile, outputDir)

    mapping_betaValues = createMappingBetaValues(betaValuesFile)
    getConsensusRegion(gffFile, bedFile, mapping_betaValues)

    plotDistribution(outputDir)
