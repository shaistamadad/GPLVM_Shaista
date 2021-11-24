
import os, torch
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sp
from collections import namedtuple

from model import GPLVM, PointLatentVariable, GaussianLikelihood, train

import matplotlib.pyplot as plt
plt.ion(); plt.style.use('ggplot')

torch.manual_seed(42)

if __name__ == '__main__':

    X_covars = np.array(pd.read_csv('data/model_mat.csv'))
    X_covars = torch.tensor(X_covars).float()[:, 2:70]

    Y = torch.tensor(sp.load_npz('data/data.npz').T.todense())
    Y /= Y.std(axis=0)

    (n, d), q = Y.shape, 6
    period_scale = np.pi

    X_latent = torch.tensor(np.load('data/init_x.npy')).float() # torch.zeros(n, q).normal_()
    X_latent = PointLatentVariable(X_latent)

    gplvm = GPLVM(n, d, q, covariate_dim=len(X_covars.T), n_inducing=len(X_covars.T) + 1, period_scale=np.pi)
    likelihood = GaussianLikelihood(batch_shape=gplvm.batch_shape)

    if torch.cuda.is_available():
        Y = Y.cuda()
        gplvm = gplvm.cuda()
        X_latent = X_latent.cuda()
        X_covars = X_covars.cuda()
        likelihood = likelihood.cuda()

    losses = train(gplvm=gplvm, likelihood=likelihood, X_covars=X_covars,
                   X_latent=X_latent, Y=Y, steps=13000, batch_size=225) # 50m

    # if os.path.isfile('model_params.pkl'):
    #     with open('model_params.pkl', 'rb') as file:
    #         state_dicts = pkl.load(file)
    #         gplvm.load_state_dict(state_dicts[0])
    #         likelihood.load_state_dict(state_dicts[1])
    #         X_latent.load_state_dict(state_dicts[2])

    with open('model_params.pkl', 'wb') as file:
        state_dicts = (gplvm.state_dict(), likelihood.state_dict(),
                       X_latent.state_dict())
        pkl.dump(state_dicts, file)

    genes = namedtuple('genes', ['CDC6', 'UBE2C', 'FN1'])
    genes = genes(CDC6=4154, UBE2C=4493, FN1=845)

    # Pseudotime plot
    t = X_latent()[:, 0].detach().cpu() % np.pi

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.scatter(t, Y[:, genes.UBE2C].detach().cpu(), alpha=0.05)
    ax1.set_title('UBE2C')
    ax2.scatter(t, Y[:, genes.CDC6].detach().cpu(), alpha=0.05)
    ax2.set_title('CDC6')
    ax3.scatter(t, Y[:, genes.FN1].detach().cpu(), alpha=0.05)
    ax3.set_title('FN1')
    ax2.set_xlabel('tau')
    ax1.set_ylabel('log(cpm/100 + 1)')

    t_bechmark = np.loadtxt('GASPACHO/Data/lv.txt')[:, 0]
    plt.scatter(t_bechmark, t, alpha=0.01)
    plt.xlabel('tau benchmark')
    plt.ylabel('tau minibatch model run')
    plt.title('Tau Comparison')
