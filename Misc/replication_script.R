
library(reticulate)
use_python('~/miniconda3/bin/python', required=T)

# Python imports

py_run_string("
import os, torch
from models.demo_model import GPLVM, train
from models.likelihoods import GaussianLikelihoodWithMissingObs
")

# Load data

Y = readRDS('GASPACHO/Data/log_cpm_4999_22188.RDS')
Y = t(as.matrix(Y)) # gc()
missing_idx = Y == 0
Y[missing_idx] = NA_real_

n = as.integer(dim(Y)[1])
d = as.integer(dim(Y)[2])
q = as.integer(6)

X_init = readRDS('GASPACHO/Data/init_param.RDS')$Xi
X_init = cbind(X_init[[1]], X_init[[2]])

model = py$GPLVM(n, d, q, n_inducing=64L, period_scale=pi, X_init=X_init)
likelihood = py$GaussianLikelihoodWithMissingObs(batch_shape=model$batch_shape)

if (py$torch$cuda$is_available()) {
    device = 'cuda'
    model = model$cuda()
    likelihood = likelihood$cuda()
} else {
    device = 'cpu'
}

Y = py$torch$tensor(Y, device=device)
losses = py$train(model, likelihood, Y, steps=10000L, batch_size=40L)

pseudotime = model$X()$cpu()$detach()$numpy()[, 1]
