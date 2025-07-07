#!/bin/bash

ENV_NAME="prep_test"

# Create conda environment with Python 3.9 (adjust version if needed)
conda create -y -n $ENV_NAME python=3.9 pip

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# PyMOL is not on conda-forge main channel, install via conda-forge
conda install -y -c conda-forge -c schrodinger pymol-bundle 

# Optional: upgrade pip and install any pip-only packages if needed
pip install --upgrade pip
pip install tqdm pandas scikit-learn argparse biotite kedro kedro-datasets
conda install numpy==1.26.4 -y
pip install torch==2.1.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_cluster --find-links https://pytorch-geometric.com/whl/torch-2.1.1+cu118.html
pip install tensorboard==2.18.0
pip install e3nn==0.5.1 # possibly not compatible with e3nn > 0.5.4
pip install scipy==1.13.1
pip install rdkit-pypi==2022.9.5
pip install openbabel-wheel==3.1.1.20
pip install biopython==1.84
pip install biotite==0.40.0
pip install atom3d
pip install wandb==0.18.2
pip install orjson

echo "Conda environment '$ENV_NAME' set up successfully."
echo "Remember to activate it with: conda activate $ENV_NAME"
