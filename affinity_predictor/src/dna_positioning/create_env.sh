#!/bin/bash

ENVNAME=dna_positioning
conda create -n $ENVNAME python=3.12 -y
conda activate $ENVNAME

conda install -c conda-forge -c schrodinger pymol-bundle -y
pip install "numpy<2"
pip install tqdm