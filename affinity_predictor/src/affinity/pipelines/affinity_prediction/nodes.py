"""
This is a boilerplate pipeline 'affinity_prediction'
generated using Kedro 0.19.12
"""

# straight foreward: call the pesto module then submmits pestos output to atomica module and run atomica

import subprocess
import csv
from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def run_python_script_in_env(script: str, env_name: str):
    command = f"""
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate {env_name}
    python {script}
    """
    subprocess.run(command, shell=True, executable="/bin/bash", check=True)


def run_bash_script_in_env(script_path: str, env_name: str):
    """
    Run a bash script inside a specified conda environment.

    Args:
        script_path (str): Path to the bash script to execute.
        env_name (str): Conda environment to activate.
    """
    command = f"""
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate {env_name}
    bash {script_path}
    """
    subprocess.run(command, shell=True, executable="/bin/bash", check=True)


def create_DNA_pdbs():
    run_python_script_in_env(
        script="../../src/dna_positioning/create_dna.py", env_name="dna_positioning"
    )


def create_complex_pdbs():
    run_python_script_in_env(
        script="../../src/dna_positioning/create_complexes_pdbs.py",
        env_name="dna_positioning",
    )


def create_data_index(folder_path: str, output_csv: str):
    pdb_entries = []
    folder_path = Path(folder_path)

    print(f"Collecting PDB data from {folder_path}...")
    for pdb_file in folder_path.rglob("*.pdb"):
        pdb_id = pdb_file.stem.lower()
        pdb_path = "../" + str(pdb_file)
        # Relevant Fields
        chain1 = "A"
        chain2 = "B_C"
        lig_code = ""
        lig_smiles = ""
        lig_resi = ""

        pdb_entries.append(
            {
                "pdb_id": pdb_id,
                "pdb_path": pdb_path,
                "chain1": chain1,
                "chain2": chain2,
                "lig_code": lig_code,
                "lig_smiles": lig_smiles,
                "lig_resi": lig_resi,
            }
        )

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "pdb_id",
            "pdb_path",
            "chain1",
            "chain2",
            "lig_code",
            "lig_smiles",
            "lig_resi",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pdb_entries)


def process_pdbs():
    run_python_script_in_env(
        script="../../src/submodules/ATOMICA/data/process_pdbs.py --data_index_file ../../../data/02_intermediate/pdb_index.csv --out_path ../../../data/02_intermediate/processed_pdbs.pkl",
        env_name="atomicaenv",
    )


def add_affinities_and_split(
    input_pkl: str, output_train_pkl: str, output_valid_pkl: str, test_size: float = 0.2
):
    # === CONFIGURATION ===
    mu = 0.0
    sigma = 125.0  # 95% of values will fall between -250 and +250
    min_affinity_value = 1e-5  # to avoid log(0) in neglog_aff computation

    # === LOAD ===
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} items from {input_pkl}")

    # === GENERATE FAKE AFFINITIES ===
    for item in data:
        raw_affinity = np.random.normal(loc=mu, scale=sigma)
        affinity_for_log = max(abs(raw_affinity), min_affinity_value)
        neglog_aff = -np.log(affinity_for_log)

        # item['label'] = float(neglog_aff) # probably unnecessary!
        item["affinity"] = {"neglog_aff": float(neglog_aff)}

    # === TRAIN-TEST SPLIT ===
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    # === SAVE SPLITS ===
    with open(output_train_pkl, "wb") as f:
        pickle.dump(train_data, f)
    with open(output_valid_pkl, "wb") as f:
        pickle.dump(test_data, f)


def run_atomica():
    run_bash_script_in_env(
        script="../../src/submodules/ATOMICA/scripts/train_atomica_affinity.sh",
        env_name="atomica",
    )
