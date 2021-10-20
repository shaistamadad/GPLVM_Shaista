import os, torch
import numpy as np
from scanpy import read_h5ad
from collections import namedtuple

from models.demo_model import GPLVM, train
from models.likelihoods import GaussianLikelihoodWithMissingObs

import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

torch.manual_seed(42)

if __name__ == '__main__':
    input_h5ad = 'GASPACHO/Data/ipsc_scRNA.h5ad'
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