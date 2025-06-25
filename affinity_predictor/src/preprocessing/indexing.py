import csv
from pathlib import Path
from tqdm import tqdm
import numpy as np

def collect_pdb_data(folder_path, output_csv="../../data/02_intermediate/pdb_index.csv"):
    pdb_entries = []
    folder_path = Path(folder_path)

    for pdb_file in tqdm(folder_path.rglob("*.pdb"), desc="Collecting PDB data"):
        pdb_id = pdb_file.stem.lower()
        pdb_path = "../" + str(pdb_file)
        # Relevant Fields
        chain1 = "A"
        chain2 = "B_C"
        lig_code = ""
        lig_smiles = ""
        lig_resi = ""

        pdb_entries.append({
            "pdb_id": pdb_id,
            "pdb_path": pdb_path,
            "chain1": chain1,
            "chain2": chain2,
            "lig_code": lig_code,
            "lig_smiles": lig_smiles,
            "lig_resi": lig_resi
        })

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["pdb_id", "pdb_path", "chain1", "chain2", "lig_code", "lig_smiles", "lig_resi"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pdb_entries)

# Example usage:
if __name__ == "__main__":
    collect_pdb_data("../../data/02_intermediate/complexes_pdbs")
