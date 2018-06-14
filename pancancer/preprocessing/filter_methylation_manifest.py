import os, sys, argparse

if __name__ == "__main__":
    # extract manifest file
    parser = argparse.ArgumentParser(
            description='Filter Manifest File for relevant Cancers')
    parser.add_argument('-m', '--manifest', help='Manifest file',
                        dest='manifest_file',
                        type=str
                        )
    parser.add_argument('-o', '--out', help='Output Manifest file',
                    dest='out_file',
                    type=str
                    )
    args = parser.parse_args()

    # cancer type list
    cancer_types = ['blca', 'brca', 'coadread', 'gbm', 'hnsc',
                    'kirc', 'laml', 'luad', 'lusc', 'ov', 'ucec']
    
    # kick out lines that do not contain our cancer types
    lines_removed = 0
    lines_kept = 0
    with open(args.manifest_file, 'r') as f:
        with open(args.out_file, 'w') as out:
            for line in f.readlines():
                dlname = line.split('\t')[1].strip()
                print (dlname)
                # get the cancer name
                if not dlname.endswith('.txt'): # header line or no filename
                    out.write(line)
                else:
                    cancer_name = dlname.split('.')[1].strip().split('_')[1].strip()
                    if cancer_name.lower() in cancer_types:
                        out.write(line)
                        lines_kept += 1
                    else:
                        lines_removed += 1
    to_print = "Removed {} files from manifest file. Kept {} files to download"
    print (to_print.format(lines_removed, lines_kept))