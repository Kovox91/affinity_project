# Overview
This pipeline offers an easy way to create data inputs for ATOMICA to predict protein-DNA affinites based on DNA sequences.  
The pipeline requires an established structure of the complex either from experimental data or an alpha fold 3 output as well as a list of DNA seqeunces to be modelled.
Using pymol functions, a DNA structure is generated from the sequences, which is subseqeuntly replacing the original DNA seqeunce in the established DNA protein complex.
All complexes are treated with the ATOMICA data preparation functions, to annotate relevant residues of the interface.
In the end the data is split into test (n=50), valid (20%) and train (80%).


## Installation
This was only tested in a Linux environment, and will probably not work under windows and mac due to the reliance on pymol.
1) **Clone the repo**  
Since we rely on a modified version of ATOMICAs functionality for the preparation of pdbs files, clone the repo with its submodules:
```
git clone --recurse-submodules https://github.com/Kovox91/affinity_project.git
```

if you already clone the repo the usual way you will need to initialize the submodules:
```
git submodule update --init --recursive
```

2) **Install dependencies**  
We recommend to install everything in a conda environment, due to the need of pymol. For that you can use the `environment_setup.sh` script.

3) **Test the setup**
If you want to be sure that everyhting was set up properly do a test run with the available data. change into the affinity_predictor folder and start the pipeline wiht `kedro run`. It will take a stored list of DNA seqeunces, generate .pdb files of the [Stereogenic Factor 1](https://www.rcsb.org/structure/2FF0) bound to the list of DNA seqeunces, and prepare the data as input with ATOMICA.