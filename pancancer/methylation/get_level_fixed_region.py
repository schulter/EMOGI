import os.path
import argparse
import numpy

#===============================================================
#read terminal parameters
#===============================================================

def args():
  args_parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
  args_parser.add_argument("-a","--gffFile",help="gff file with miRNA annotation",type=str,required=True)
  args_parser.add_argument("-b","--bedFile",help="bed file with annotation of DNA methylation regions",type=str,required=True)
  args_parser.add_argument("-v","--betaValuesFile",help="file with beta values and ID of DNA methylation",type=str,required=True)
  args_parser.add_argument("-o","--outputDir",help="output directory for bed analysis files (not required): default: ./",type=str, default="./")
  return args_parser.parse_args()

#===============================================================
#Methods
#===============================================================

def proofFile(filename):
  if not (os.path.exists(filename)):
    exit("{0} does not exist!".format(filename))
    return False
  if not (os.path.isfile(filename)):
    exit("{0} is not a file!".format(filename))
    return False
  return True

#===============================================================

def setOutputFilenames(bedFile,gffFile,outputDir):
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

#===============================================================

def runBedAnalysis(gffFile,bedFile,bedAnalysisFile):
  bedCmd = "intersectBed -wo -a " + gffFile + " -b " + bedFile + " > " + bedAnalysisFile
  print "\nBed Tool command: ",bedCmd
  os.system(bedCmd)

#===============================================================

def createMappingBetaValues(betaValuesFile):
  readhandle = open(betaValuesFile)
  mapping_betaValues = {}
  for line in readhandle:
    line = line.rstrip().split('\t')
    betaValue = float(line[1])
    mapping_betaValues[str(line[0])] = betaValue
  return mapping_betaValues
 
 #===============================================================
 
def caluculateDNAmethLevel(gffFile,mapping_betaValues):
  readhandle = open(bedAnalysisFile)
  IDtoDNAmeth = {}
  for line in readhandle:
    line = line.rstrip().split('\t')
    DNAmeth_ID = line[12]
    TSS_ID = line[3] + line[8]
    if mapping_betaValues.has_key(str(DNAmeth_ID)):
      betaValue = mapping_betaValues[str(DNAmeth_ID)]

      if IDtoDNAmeth.has_key(TSS_ID):
        IDtoDNAmeth[TSS_ID].append(betaValue)
      if not IDtoDNAmeth.has_key(TSS_ID):
        IDtoDNAmeth[TSS_ID] = [betaValue]

  readhandle.close()
  readhandle = open(gffFile)


  DNAmeth_coverage = []
  CpG_prom = []

  with open(DNAmethFile,'w') as writehandle:
    for line in readhandle:
		if not line.startswith('#'):
			line = line.rstrip().split('\t')
			promID = line[3] + line[8]
	
			if IDtoDNAmeth.has_key(promID): 
				DNAmeth_coverage = float(round(numpy.median(IDtoDNAmeth[promID]),3))
				CpG_prom = 'CpG'
				
				beta_values = IDtoDNAmeth[promID]
				unmeth = 0
				meth = 0
				for beta_value in beta_values:
					if beta_value <= 0.2:
						unmeth = unmeth + 1
					if beta_value >= 0.6:
						meth = meth + 1
				
				#writehandle.write('\t'.join(line) + '\t' + str(DNAmeth_coverage) + '\t' + CpG_prom + '\n')
				writehandle.write('\t'.join(line) + '\t' + str(DNAmeth_coverage) + '\t' + CpG_prom + '\t' + str(meth) + '\t' + str(unmeth) + '\t' + str(len(beta_values)) + '\n')

			else:
				DNAmeth_coverage = float(0)
				CpG_prom = 'unknown'

				#writehandle.write('\t'.join(line) + '\t' + str(DNAmeth_coverage) + '\t' + CpG_prom + '\n')
				writehandle.write('\t'.join(line) + '\t' + str(DNAmeth_coverage) + '\t' + CpG_prom + '\t0\t0\t0\n')
 
#===============================================================

def plotDistribution(outputDir):
  rCmd = "R CMD BATCH --slave \"--args " + outputDir + " " + bedAnalysisFileName + "\"  plotDistribution.R reportPlotDistribution.txt" 
  os.system(rCmd)
  print "Plotting done!"
  
#===============================================================    
#Parameters   
#===============================================================    

parameters = args()

bedFile = parameters.bedFile
proofFile(bedFile)

gffFile = parameters.gffFile
proofFile(gffFile)

betaValuesFile = parameters.betaValuesFile
proofFile(betaValuesFile)

outputDir = parameters.outputDir

#===============================================================
#executive part
#===============================================================

if not outputDir.endswith("/"):
  outputDir = outputDir + "/"

setOutputFilenames(bedFile,gffFile,outputDir)
runBedAnalysis(gffFile,bedFile,bedAnalysisFile)
mapping_betaValues = createMappingBetaValues(betaValuesFile)
caluculateDNAmethLevel(gffFile,mapping_betaValues)
plotDistribution(outputDir)
