#!/usr/bin/env python
# coding: utf-8
# %%

# %%

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
sys.path.append('/home/jupyter/BGPLVM_scRNA/')
import model
from model import GPLVM, PointLatentVariable, GaussianLikelihood, train
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%

def run_model(adata):
    '''
    Wrapper function to train fast GPLVM model on single cell datasets
    
    Params:
    ------
    - adata: anndata object with the gene counts
    
    Returns:
  
    - anndata object storing trained GPVLM model in adata.obsm
    '''
    
    #Preprocessing and choosing the most variable genes to make the traning efficient 
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    adata = adata[:, adata.var['highly_variable']].copy()  #slicing of anndata object needs copy 
    # scale the data with scanpy.pp.scale with zero_center= TRUE or false; after scaling anndata     
    #becomes a dense object 
    #save layers of data: anndata.layers (save different versions of the matrix)
    sc.tl.pca(adata, svd_solver='arpack')
    
    # Save the first 7 PC components as X_init variable to initialise the latent variables
    n_dim = 6
    n_cells=adata.shape[0]
    X_init=adata.obsm['X_pca'][0:n_cells,0:n_dim+1]
    X_init=torch.tensor(X_init)
    X_init=X_init.float() 
    
    
    #X_covars = np.array(pd.read_csv('data/model_mat.csv'))
    #X_covars = torch.tensor(X_covars).float()[:, 2:70]
    
    #Y = torch.tensor(sp.load_npz('/home/jupyter/gastrulationdata.npz').todense())
    Y=torch.tensor((adata.X).todense())
    Y /= Y.std(axis=0)  #potentially not do this 

    (n, d), q = Y.shape, n_dim
    period_scale = np.pi
    use_pseudotime=True


    #X_latent = torch.zeros(n, q+use_pseudotime).normal_() #torch.tensor(np.load('/home/jupyter/mount/gdrive/BGPLVM_scRNA/init_x.npy')).float() # torch.zeros(n, q).normal_()
    X_latent= X_init
    X_latent = PointLatentVariable(X_latent)
    X_covars = torch.zeros(n, 0)
   

    gplvm = GPLVM(
        n, d, q, covariate_dim=len(X_covars.T), n_inducing=60, pseudotime_dim=use_pseudotime, period_scale=np.pi)
    likelihood = GaussianLikelihood(batch_shape=gplvm.batch_shape)
    
    if torch.cuda.is_available():
        Y = Y.cuda()
        gplvm = gplvm.cuda()
        X_latent = X_latent.cuda()
        X_covars = X_covars.cuda()
        likelihood = likelihood.cuda()
        
    #Train the Model

    losses = train(gplvm=gplvm, likelihood=likelihood, X_covars=X_covars,
                   X_latent=X_latent, Y=Y, steps=13000, batch_size=225) # 50m
    
    #save training outputs in adata 
    
    adata.uns['model_state_dict'] = gplvm.state_dict()
    for key,value in adata.uns['model_state_dict'].items():
        if torch.is_tensor(value):
            adata.uns['model_state_dict'][key] = value.cpu().detach().numpy()
            
    adata.uns['likelihood_state_dict'] = likelihood.state_dict()
    for key,value in adata.uns['likelihood_state_dict'].items():
        if torch.is_tensor(value):
            adata.uns['likelihood_state_dict'][key] = value.cpu().detach().numpy()

    #Save pseudotime 
    t = X_latent()[:, 0].cpu().detach().numpy()
    adata.obs['cellcycle_pseudotime'] = t
    ## Store latent dimensions
    X_latent = X_latent()[:, 1:].cpu().detach()
    X_latent=X_latent.cpu().detach().numpy()
    adata.obsm['X_BGPLVM_latent'] = X_latent
    
    #Test
    if adata.obsm['X_BGPLVM_latent'].shape[1] != n_dim:
         print("ERROR! The number of dimensions is wrong")
    if all(adata.obs['cellcycle_pseudotime'] == adata.obsm['X_BGPLVM_latent'][:,0]):
        print("ERROR! You saved the wrong cellcycle latent variable")

    trained_adata= adata.copy()
    
    return trained_adata





# %%


def scatter_Plot(adata):
    '''
   
    '''
    
    ## Get 1st smooth latent variable (not periodic i.e. not cellcycle)
    adata.obs['GPLVM_LV1'] = adata.obsm["X_BGPLVM_latent"][:,0]
    sc.pl.scatter(adata, 'GPLVM_LV1', 'cellcycle_pseudotime')
    return


# %%


def plot_cormap(adata,PCs):
    '''
    show a correlation heatmap between the latent variables (excluding the cell cycle and the PCs
    '''
    X_lv = adata.obsm["X_BGPLVM_latent"].copy()
    X_pc = adata.obsm["X_pca"].copy()

    n_pcs = PCs
    n_gplvm_dims = X_lv.shape[1]
    
    cormat = np.corrcoef(X_lv.T, X_pc[:,0:n_pcs].T)

    pcVSlv_cormat = cormat[0:n_gplvm_dims,n_gplvm_dims:n_gplvm_dims+n_pcs]
    sns.heatmap(pcVSlv_cormat, cmap="RdBu_r", vmax=1, vmin=-1);
    plt.xlabel('PCs');
    plt.ylabel("GPLVM Variables")
    
    return 


# %%

def run_model_randomInit(adata):
    '''
    Wrapper function to train fast GPLVM model on single cell datasets
    
    Params:
    ------
    - adata: anndata object with the gene counts
    -X_init is a random initialisation 
    
    Returns:
  
    - anndata object storing trained GPVLM model in adata.obsm
    '''
    
    #Preprocessing and choosing the most variable genes to make the traning efficient 
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.tl.pca(adata, svd_solver='arpack')
    n_dim = 6
    #X_covars = np.array(pd.read_csv('data/model_mat.csv'))
    #X_covars = torch.tensor(X_covars).float()[:, 2:70]
    
    #Y = torch.tensor(sp.load_npz('/home/jupyter/gastrulationdata.npz').todense())
    Y=torch.tensor((adata.X).todense())
    Y /= Y.std(axis=0)

    (n, d), q = Y.shape, n_dim
    period_scale = np.pi
    use_pseudotime=True


    X_latent = torch.zeros(n, q+use_pseudotime).normal_() #torch.tensor(np.load('/home/jupyter/mount/gdrive/BGPLVM_scRNA/init_x.npy')).float() # torch.zeros(n, q).normal_()
    X_latent = PointLatentVariable(X_latent)
    X_covars = torch.zeros(n, 0)
   

    gplvm = GPLVM(
        n, d, q, covariate_dim=len(X_covars.T), n_inducing=60, pseudotime_dim=use_pseudotime, period_scale=np.pi)
    likelihood = GaussianLikelihood(batch_shape=gplvm.batch_shape)
    
    if torch.cuda.is_available():
        Y = Y.cuda()
        gplvm = gplvm.cuda()
        X_latent = X_latent.cuda()
        X_covars = X_covars.cuda()
        likelihood = likelihood.cuda()
        
    #Train the Model

    losses = train(gplvm=gplvm, likelihood=likelihood, X_covars=X_covars,
                   X_latent=X_latent, Y=Y, steps=13000, batch_size=225) # 50m
    
    #save training outputs in adata 
    
    adata.uns['model_state_dict'] = gplvm.state_dict()
    for key,value in adata.uns['model_state_dict'].items():
        if torch.is_tensor(value):
            adata.uns['model_state_dict'][key] = value.cpu().detach().numpy()
            
    adata.uns['likelihood_state_dict'] = likelihood.state_dict()
    for key,value in adata.uns['likelihood_state_dict'].items():
        if torch.is_tensor(value):
            adata.uns['likelihood_state_dict'][key] = value.cpu().detach().numpy()

    #Save pseudotime 
    t = X_latent()[:, 0].cpu().detach().numpy()
    adata.obs['cellcycle_pseudotime'] = t
    ## Store latent dimensions
    X_latent = X_latent()[:, 1:].cpu().detach()
    X_latent=X_latent.cpu().detach().numpy()
    adata.obsm['X_BGPLVM_latent'] = X_latent
    
    #Test
    if adata.obsm['X_BGPLVM_latent'].shape[1] != n_dim:
         print("ERROR! The number of dimensions is wrong")
    if all(adata.obs['cellcycle_pseudotime'] == adata.obsm['X_BGPLVM_latent'][:,0]):
        print("ERROR! You saved the wrong cellcycle latent variable")

    trained_adata= adata.copy()
    
    return trained_adata




