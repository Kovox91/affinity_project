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

def generate_dna_pdb(seq: str) -> str:
    cmd.fnab(seq, "dna")
    result = cmd.get_pdbstr("dna")
    cmd.delete("dna")
    return result

def create_DNA_pdbs(sequences: list[str]) -> dict[str, str]:
    output = {}
    for seq in tqdm(sequences, desc="Generating PDBs"):
        pdb_str = generate_dna_pdb(seq)
        output[seq] = pdb_str
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
        new_chain = model_m[chain_id_mutant].copy()
        new_chain.id = new_chain_id
        model_c.add(new_chain)

def create_complex_pdbs(mutant_structures: dict[str, str], complex_template: str) -> dict[str, str]:
    parser = PDBParser(QUIET=True)
    complex_structure = parser.get_structure("complex", StringIO(complex_template))[0]

    result = {}

    for filename, mutant_pdb in tqdm(mutant_structures.items(), desc="Creating Complex PDBs"):
        mutant_structure = parser.get_structure("mutant", StringIO(mutant_pdb))[0]

        complex_res_B = set(get_residues_with_full_backbone(complex_structure, "B", RESIDUES_COMPLEX))
        mutant_res_A  = set(get_residues_with_full_backbone(mutant_structure, "A", RESIDUES_MUTANT))

        complex_atoms_B = get_backbone_atoms_filtered(complex_structure, "B", RESIDUES_COMPLEX)
        mutant_atoms_A  = get_backbone_atoms_filtered(mutant_structure, "A", RESIDUES_MUTANT)

        if len(complex_atoms_B) != len(mutant_atoms_A):
            raise ValueError(f"Backbone atom counts differ in {filename}: {len(complex_atoms_B)} vs {len(mutant_atoms_A)}")

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

def create_data_index(pdb_files: dict[str, str]) -> pd.DataFrame:
    pdb_entries = []

    for filename in tqdm(pdb_files.keys(), desc="Indexing PDBs"):
        pdb_id = filename.replace(".pdb", "").lower()
        pdb_path = f"../data/02_intermediate/complexes_pdbs/{filename}"

        pdb_entries.append({
            "pdb_id": pdb_id,
            "pdb_path": pdb_path,
            "chain1": "A",
            "chain2": "B_C",
            "lig_code": "",
            "lig_smiles": "",
            "lig_resi": ""
        })

    return pd.DataFrame(pdb_entries)

def process_pdbs(pdb_index: pd.DataFrame, params: dict) -> list[dict]:
    return process_all_pdbs(
        index_df=pdb_index,
        dist_th=params.get("interface_dist_th", 8.0),
        fragmentation_method=params.get("fragmentation_method", None)
    )

def add_affinities(
    items: list[dict],
    params: dict
) -> tuple[list[dict], list[dict]]:
    mu = params.get("mu", 0.0)
    sigma = params.get("sigma", 125.0)
    min_affinity_value = params.get("min_affinity_value", 1e-5)
    test_size = params.get("test_size", 0.2)
    seed = params.get("random_state", 42)

    for item in items:
        raw_affinity = np.random.normal(loc=mu, scale=sigma)
        affinity_for_log = max(abs(raw_affinity), min_affinity_value)
        neglog_aff = -np.log(affinity_for_log)
        item["affinity"] = {"neglog_aff": float(neglog_aff)}

    train_data, test_data = train_test_split(items, test_size=test_size, random_state=seed)
    return train_data, test_data

def run_atomica():
    pass