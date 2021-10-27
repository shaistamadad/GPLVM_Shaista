# GPLVM_Shaista
This is the repo for rotation in Teichmann Lab at Sanger Institute under the supervision of Emma Dann. 

# Conda environment setup
To create new conda environment with required packages for training the GPLVM model, run in terminal:
```bash
conda create -n gplvm_env python=3.7 
conda activate gplvm_env 
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install prettytable ipython numpy gpytorch==1.4.1 scanpy ipykernel
```
To add the conda environment to the list of kernels available to jupyterLab, run:
```bash
python -m ipykernel install --user --name gplvm_env --display-name "Environment (gplvm_env)"
```
Then reload the browser tab of JupyterLab to see the new environment in the list of available kernels.
