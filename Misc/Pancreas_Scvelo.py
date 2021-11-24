#!/usr/bin/env python
# coding: utf-8

# First install and import all the requires packages 

# First install the necessary packages

# In[88]:


#pip install -U scvelo


# In[ ]:


#conda install pytorch torchvision -c pytorch


# In[ ]:


#pip install python-igraph louvain


# In[ ]:


#pip install pybind11 hnswlib


# In[ ]:


#pip install gpytorch==1.4.1


# In[28]:


#gpytorch.__version__ #downgrade


# In[1]:


import scvelo as scv


# In[2]:


import os, torch
import numpy as np
from scanpy import read_h5ad
from collections import namedtuple


# In[27]:


import gpytorch


# In[3]:


import sys  #sys and os 
cwd = './BGPLVM_scRNA/' #madd BGPLVM to path so the models module can be imported easily 
sys.path.append(cwd)
import models  #good practice to make the first chunk to import all the packages you need to avoid import calls in the middle of the script 


# In[4]:


from models.demo_model import GPLVM, train


# In[6]:


from models.demo_model import GPLVM


# In[7]:


from models.likelihoods import GaussianLikelihoodWithMissingObs


# In[8]:


import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

torch.manual_seed(42)


# Import the pancreas dataset from the scVelo package 

# In[9]:


pancreasData=scv.datasets.pancreas()


# In[10]:


pancreasData


# Run the main code on the ipsc dataset 

# In[11]:


if __name__ == '__main__':
    input_h5ad = '/home/jovyan/mount/gdrive/rotation1/ipsc_scRNA.h5ad'
    adata = read_h5ad(input_h5ad)

    Y = adata.X.copy()
    missing_loc = ~(Y != 0).todense()
    Y = Y.todense()
    Y[missing_loc] = np.nan

    (n, d), q = Y.shape, 6

    model = GPLVM(n, d, q, n_inducing=64, period_scale=np.pi,
                   X_init=adata.obsm["X_init"])
    likelihood = GaussianLikelihoodWithMissingObs(batch_shape=model.batch_shape)

    if torch.cuda.is_available():
        device = 'cuda'
        model = model.cuda()
        likelihood = likelihood.cuda()
    else:
        device = 'cpu'

   


# In[12]:


torch.cuda.is_available()


# In[13]:


Y = torch.tensor(Y, device=device)


# In[14]:


Y


# In[ ]:


losses = train(model, likelihood, Y, steps=10000, batch_size=40)


# In[ ]:


Y = torch.tensor(Y, device=device)
   losses = train(model, likelihood, Y, steps=10000, batch_size=40)

   ## Store model 
   adata.uns['model_state_dict'] = model.state_dict()
   adata.uns['likelihood_state_dict'] = likelihood.state_dict()

   ## Store latent dimensions
   X_latent = model.X()[:, 1:].cpu().detach()
   adata.obsm['X_BGPLVM_latent'] = X_latent

   ## Store cellcycle pseudotime
   t = model.X()[:, 0].cpu().detach()
   adata.obs['cellcycle_pseudotime'] = t

   adata.write_h5ad('{p}.trainedBGPLVM.h5ad'.format(p=input_h5ad.split('.h5ad')[0]))

   genes = namedtuple('genes', ['CDC6', 'UBE2C', 'FN1'])
   genes = genes(CDC6=4154, UBE2C=4493, FN1=845)

   # Pseudotime plot
   plt.scatter(t, Y[:, genes.FN1].detach().cpu(), alpha=0.05)
   plt.title('FN1')
   plt.xlabel('tau')
   plt.ylabel('log(cpm/100 + 1)')

   x_fwd = torch.zeros(100, q+1)
   x_fwd[:, 0] = torch.linspace(0, model.X.period_scale, 100)
   y_fwd = model.cpu()(x_fwd, batch_idx=torch.arange(100)).loc[genes.FN1, :].detach()
   plt.plot(x_fwd[:, 0], y_fwd)

   # # Latent dims plot
   # plt.scatter(X[:, 1], X[:, 2], alpha=0.01)
   # plt.xlabel('Latent Dim 1')
   # plt.ylabel('Latent Dim 2')

   t_bechmark = np.loadtxt('GASPACHO/Data/lv.txt')[:, 0]
   plt.scatter(t_bechmark, t, alpha=0.01)
   plt.xlabel('tau benchmark')
   plt.ylabel('tau minibatch model run')
   plt.title('Tau Comparison')


# In[ ]:




