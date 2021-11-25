# GPLVM_Shaista
This is the repo for rotation in Teichmann Lab at Sanger Institute under the supervision of Emma Dann. 

# Conda environment setup
To create new conda environment with required packages for training the GPLVM model, run in terminal:
```bash
conda create -n gplvm_env python=3.7 
conda activate gplvm_env 
pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install prettytable ipython numpy gpytorch==1.4.1 scanpy ipykernel  #gpytorch version 1.5.1?
```
To add the conda environment to the list of kernels available to jupyterLab, run:
```bash
python -m ipykernel install --user --name gplvm_env --display-name "Environment (gplvm_env)" 
```
Then reload the browser tab of JupyterLab to see the new environment in the list of available kernels.


Model document is at https://www.overleaf.com/project/605e07104b6c6221cfa7a557

# Code Organisation:

The file TrainFunction.py contains two functions: run_model and run_model_randomInit to train an anndata object with 1 to 7 PC components or random values values. TrainFunction.py requires GPLVM model classes from file model.py.

model.py and demo_re_as_linear.py: copied from Aditya-edits branch of BGPLVM repo: https://github.com/vr308/BGPLVM_scRNA


run_model.py: python script to train GPLVM model for any anndata object: 

usage: python run_model.py input initialisation output 
input: choose one of six dataset options (bonemarrow, gastrulation, forebrain, pancreas, pbmc, iPSC) or the path to your own anndata object.
initialisation: choose one of two options: random or PCA
output: name/path to output anndata file with trained GPLVM model