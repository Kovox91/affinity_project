#!/bin/bash

ENVNAME=dna_positioning

# Source conda.sh to enable conda activate in the script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create environment if it doesn't exist
if ! conda info --envs | grep -q "^$ENVNAME"; then
    conda create -n "$ENVNAME" python=3.12 -y
fi

# Activate environment
conda activate "$ENVNAME"

# Check if activation succeeded before proceeding
if [ $? -eq 0 ]; then
    conda install -c conda-forge -c schrodinger pymol-bundle -y
    pip install "numpy<2"
    pip install tqdm
    pip install biopython
else
    echo "Failed to activate conda environment: $ENVNAME"
    exit 1
fi
