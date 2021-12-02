#!/usr/bin/env python
# coding: utf-8

from TrainFunction import run_model, run_model_randomInit
import scvelo as scv
import scipy
import os, torch
import numpy as np
from scanpy import read_h5ad
from collections import namedtuple
import csv
import gpytorch
import scanpy as sc
import scipy.sparse as sp
import sys  #sys and os 
sys.path.append('/home/jupyter/BGPLVM_scRNA/') #append the path to the repo containing GPLVM training functions (model: GPVLM, PointLatentVariable, GaussainLikelihood, Train
import model
from model import GPLVM, PointLatentVariable, GaussianLikelihood, train

# for plots 
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

#for specifying inputs and outputs from the commandline 
import argparse

parser= argparse.ArgumentParser(description='This script take an anndata object as input and trains the GPLVM model on the input returning an anndata object with the latent variable embedding ')
parser.add_argument('input', type=str,
                 help=' the input anndata object.Choose one of the six  options: bonemarrow, forebrain, gastrulation,iPSC, pancreas,pbmc10k. Or else give the path to the anndata object you want to train' )

parser.add_argument('initialisation', type= str,
                 help='choose either random or PCA as the initialisation for the latent variable matrix')

parser.add_argument('output', type=str,
                 help='name of outputfile containing the trained GPLVM model')


args = parser.parse_args()


if args.input == 'bonemarrow': 
    adata= scv.datasets.bonemarrow() 

if args.input == 'forebrain': 
    adata= scv.datasets.forebrain() 

if args.input == 'gastrulation': 
    adata= scv.datasets.gastrulation()  

if args.input == 'iPSC': 
    input_h5ad = '/home/jupyter/mount/gdrive/BGPLVM_scRNA/ipsc_scRNA.h5ad'
    adata = read_h5ad(input_h5ad)

if args.input == 'pancreas': 
    adata= scv.datasets.pancreas()  

if args.input == 'pbmc10k': 
    adata = read_h5ad('/home/jupyter/mount/gdrive/BridgeIntegrationData/Data/pbmc10k.h5ad')

 

 # else: 
#     input_h5ad = args.input
#     adata = read_h5ad(input_h5ad)

#print(type(args.input))

torch.manual_seed(42)
if args.initialisation=='random':
    
    Trained=run_model_randomInit(adata)
    Trained.write_h5ad('/home/jupyter/GPLVM_Shaista/TrainedModels/' + args.output + '.h5ad') 

if   args.initialisation=='PCA':

    Trained=run_model(adata)
    Trained.write_h5ad('/home/jupyter/GPLVM_Shaista/TrainedModels/' + args.output + '.h5ad')




