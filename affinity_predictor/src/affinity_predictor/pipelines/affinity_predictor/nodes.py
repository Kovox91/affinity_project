"""
This is a boilerplate pipeline 'affinity_prediction'
generated using Kedro 0.19.12
"""

# straight foreward: call the pesto module then submmits pestos output to atomica module and run atomica

from tqdm import tqdm
from pymol import cmd
from Bio.PDB import PDBParser, PDBIO, Superimposer
from io import StringIO
import pandas as pd
from submodules.ATOMICA.data.process_pdbs import process_all_pdbs
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

def generate_dna_pdb(seq: str) -> str:
    cmd.fnab(seq, "dna")
    result = cmd.get_pdbstr("dna")
    cmd.delete("dna")
    return result


def create_DNA_pdbs(sequence_file_content: str) -> dict[str, str]:
    output = {}
    lines = sequence_file_content.strip().split("\n")
    for line in tqdm(lines, desc="Generating DNA PDBs"):
        if len(line.split()) != 5:
            warnings.warn(f"Skipping line due to missing values (expected 5 columns): {line}")
            continue
        sequence = line.split()[0]  # Only take the first element
        pdb_str = generate_dna_pdb(sequence)
        output[sequence] = pdb_str
    return output

BACKBONE_ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]
RESIDUES_COMPLEX = list(range(2, 16))
RESIDUES_MUTANT = list(range(5, 19))


def get_residues_with_full_backbone(structure, chain_id, allowed_residues):
    full_res = []
    for residue in structure[chain_id]:
        if residue.id[1] not in allowed_residues:
            continue
        if all(atom in residue for atom in BACKBONE_ATOMS):
            full_res.append(residue.id[1])
    return full_res


def get_backbone_atoms_filtered(structure, chain_id, residue_ids):
    atoms = []
    for residue in structure[chain_id]:
        if residue.id[1] not in residue_ids:
            continue
        if all(atom in residue for atom in BACKBONE_ATOMS):
            for atom_name in BACKBONE_ATOMS:
                atoms.append(residue[atom_name])
    return atoms


def replace_chain(complex_struct, mutant_struct, chain_id_complex, chain_id_mutant, new_chain_id):
    model_c = complex_struct
    model_m = mutant_struct
    
    if chain_id_complex in model_c:
        model_c.detach_child(chain_id_complex)
    if chain_id_mutant in model_m:
        # Copy and rename the mutant chain
        new_chain = model_m[chain_id_mutant].copy()
        new_chain.id = new_chain_id
        model_c.add(new_chain)



def create_complex_pdbs(
    mutant_structures: dict[str, str], complex_template: str
) -> dict[str, str]:
    parser = PDBParser(QUIET=True)

    result = {}

    for filename, mutant_pdb in tqdm(
        mutant_structures.items(), desc="Creating Complex PDBs"
    ):
        complex_structure = parser.get_structure("complex", StringIO(complex_template))[0]
        mutant_structure = parser.get_structure("mutant", StringIO(mutant_pdb))[0]

    
        complex_atoms_B = get_backbone_atoms_filtered(
            complex_structure, "B", RESIDUES_COMPLEX
        )
        mutant_atoms_A = get_backbone_atoms_filtered(
            mutant_structure, "A", RESIDUES_MUTANT
        )

        if len(complex_atoms_B) != len(mutant_atoms_A):
            raise ValueError(
                f"Backbone atom counts differ in {filename}: {len(complex_atoms_B)} vs {len(mutant_atoms_A)}"
            )

        sup = Superimposer()
        sup.set_atoms(complex_atoms_B, mutant_atoms_A)
        sup.apply(mutant_structure.get_atoms())

        replace_chain(complex_structure, mutant_structure, "B", "A", "B")
        replace_chain(complex_structure, mutant_structure, "C", "B", "C")

        io = PDBIO()
        output_io = StringIO()
        io.set_structure(complex_structure)
        io.save(output_io)

        result[filename] = output_io.getvalue()

    return result


def save_pdb_files(pdb_dict: dict[str, str], output_folder: str) -> None:
    import os
    os.makedirs(output_folder, exist_ok=True)
    for filename, pdb_str in pdb_dict.items():
        if not filename.endswith(".pdb"):
            filename += ".pdb"
        filepath = os.path.join(output_folder, filename)
        with open(filepath, "w") as f:
            f.write(pdb_str)


def create_data_index(
    pdb_files: dict[str, str],
    output_folder: str
) -> pd.DataFrame:
    pdb_entries = []

    for filename in tqdm(pdb_files.keys(), desc="Indexing PDBs"):
        pdb_id = filename.replace(".pdb", "").lower()
        pdb_path = f"{output_folder}/{filename}.pdb"
        pdb_entries.append(
            {
                "pdb_id": pdb_id,
                "pdb_path": pdb_path,
                "chain1": "A",
                "chain2": "B_C",
                "lig_code": "",
                "lig_smiles": "",
                "lig_resi": "",
            }
        )

    return pd.DataFrame(pdb_entries)


def process_pdbs(pdb_index: pd.DataFrame) -> list[dict]:
    return process_all_pdbs(
        index_df=pdb_index,
        dist_th=10.0
    )


def add_affinities(items: list[dict], affinities: str) -> tuple[list[dict], list[dict]]:
    seed = 42
    test_size = 0.2

    lines = affinities.strip().split("\n")
    valid_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            valid_lines.append(parts)
        else:
            warnings.warn(f"Skipping invalid affinity line (missing Values): {line}")

    if len(valid_lines) != len(items):
        raise ValueError(
            f"Number of valid affinity lines ({len(valid_lines)}) does not match number of items ({len(items)})"
        )

    for item, parts in zip(items, valid_lines):
        try:
            col2 = float(parts[1])
            col3 = float(parts[3])
            affinity = col3 - col2
            if affinity <= 0:
                raise ValueError(f"Non-positive affinity value ({affinity}) in line: {' '.join(parts)}")
            neglog_aff = -np.log(affinity)
            item["affinity"] = {
                "value": affinity,
                "neglog_aff": float(neglog_aff),
            }
        except Exception as e:
            raise ValueError(f"Error processing line: {' '.join(parts)}\n{e}")

    # Set aside exactly 15 items for test
    test_size = 15
    assert len(items) > test_size, "Not enough items to allocate 15 test samples"

    # First split: remove test set
    remaining_items, test_data = train_test_split(items, test_size=test_size, random_state=seed)

    # Second split: train and validation from remaining
    train_data, val_data = train_test_split(
        remaining_items, test_size=0.2, random_state=seed  # 20% validation of remaining
    )

    return train_data, test_data, val_data
