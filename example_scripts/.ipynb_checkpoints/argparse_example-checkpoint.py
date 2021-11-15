## Example python script using argparse 
# (here I read an .h5ad object and rename the column storing the celltype information)
import numpy as np
import pandas as pd
import scanpy

import argparse
parser = argparse.ArgumentParser()
## Example required argument
parser.add_argument("filepath", help="path to anndata file")
## Example optional argument
parser.add_argument("--celltype_column", 
                    default="celltype",
                    help="string indicating column in adata.obs storing cell type information")
args = parser.parse_args()

## Parse arguments
h5ad_file = args.filepath
celltype_col = args.celltype_column

## Your script starts here ##
adata = sc.read_h5ad(h5ad_file)
adata.obs['annotation'] = adata.obs[celltype_col]
output_h5ad_file = h5ad_file.split('.h5ad')[0] + ".cleaned.h5ad"
sc.write_h5ad(output_h5ad_file)
