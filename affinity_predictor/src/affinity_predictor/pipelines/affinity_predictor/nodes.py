from tqdm import tqdm
from pymol import cmd
from Bio.PDB import PDBParser, PDBIO, Superimposer
from io import StringIO
import pandas as pd
from submodules.ATOMICA.data.process_pdbs import process_all_pdbs
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import os


def generate_dna_pdb(seq: str) -> str:
    """
    Generates a PDB-format string of a DNA molecule from a nucleotide sequence.

    Args:
        seq (str): DNA sequence (e.g., "ATCG").

    Returns:
        str: PDB-formatted string representing the DNA structure.
    """
    cmd.fnab(seq, "dna")
    result = cmd.get_pdbstr("dna")
    cmd.delete("dna")
    return result


def create_DNA_pdbs(sequence_file_content: str) -> dict[str, str]:
    """
    Parses DNA sequences from input text and generates corresponding PDB-format strings, only if the input file contains required affinitie values (see README)

    Args:
        sequence_file_content (str): Multiline string where each line contains at least a DNA sequence
                                     followed by four additional columns.

    Returns:
        dict[str, str]: Mapping from DNA sequences to their corresponding PDB-formatted strings.
    """
    output = {}
    lines = sequence_file_content.strip().split("\n")
    for line in tqdm(lines, desc="Generating DNA PDBs"):
        if len(line.split()) != 5:
            warnings.warn(
                f"Skipping line due to missing values (expected 5 columns): {line}"
            )
            continue
        sequence = line.split()[0]  # Only take the first element
        pdb_str = generate_dna_pdb(sequence)
        output[sequence] = pdb_str
    return output


def get_backbone_atoms_filtered(structure, chain_id, residue_ids, backbone_atoms):
    """
    Retrieves specified backbone atoms from selected residues in a given chain of a structure.

    Args:
        structure: Structure object containing chains and residues.
        chain_id: Identifier of the chain to search within.
        residue_ids: Iterable of residue sequence IDs to filter.
        backbone_atoms: List of atom names to extract from each residue.

    Returns:
        list: List of atom objects corresponding to the specified backbone atoms.
    """
    atoms = []
    for residue in structure[chain_id]:
        if residue.id[1] not in residue_ids:
            continue
        if all(atom in residue for atom in backbone_atoms):
            for atom_name in backbone_atoms:
                atoms.append(residue[atom_name])
    return atoms


def replace_chain(
    complex_struct, mutant_struct, chain_id_complex, chain_id_mutant, new_chain_id
):
    """
    Replaces a chain in a complex structure with a chain from a mutant structure, optionally renaming it.

    Args:
        complex_struct: Structure containing the original complex.
        mutant_struct: Structure containing the mutant chain.
        chain_id_complex: Chain ID to be replaced in the complex.
        chain_id_mutant: Chain ID of the mutant chain to insert.
        new_chain_id: New chain ID to assign to the inserted mutant chain.

    Returns:
        None
    """
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
    mutant_structures: dict[str, str], complex_template: str, params: dict
) -> dict[str, str]:
    """
    Generates complex PDB structures by superimposing mutant chains onto a complex template and replacing specified chains.

    Args:
        mutant_structures (dict[str, str]): Mapping of filenames to mutant PDB strings.
        complex_template (str): PDB string of the complex template structure.
        params (dict): Parameters including residue IDs and backbone atom names:
            - "residue_complex": Residue IDs for complex chain filtering.
            - "residues_mutant": Residue IDs for mutant chain filtering.
            - "backbone_atoms": List of backbone atom names to use for superimposition.

    Returns:
        dict[str, str]: Mapping of filenames to newly created complex PDB strings.
    """
    parser = PDBParser(QUIET=True)

    result = {}

    RESIDUES_COMPLEX = params["residues_complex"]
    RESIDUES_MUTANT = params["residues_mutant"]
    BACKBONE_ATOMS = params["backbone_atoms"]

    for filename, mutant_pdb in tqdm(
        mutant_structures.items(), desc="Creating Complex PDBs"
    ):
        complex_structure = parser.get_structure("complex", StringIO(complex_template))[
            0
        ]
        mutant_structure = parser.get_structure("mutant", StringIO(mutant_pdb))[0]

        complex_atoms_B = get_backbone_atoms_filtered(
            complex_structure, "B", RESIDUES_COMPLEX, BACKBONE_ATOMS
        )
        mutant_atoms_A = get_backbone_atoms_filtered(
            mutant_structure, "A", RESIDUES_MUTANT, BACKBONE_ATOMS
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
    """
    Saves PDB strings to files in a specified output folder, ensuring the folder exists.

    Args:
        pdb_dict (dict[str, str]): Mapping of filenames to PDB-formatted strings.
        output_folder (str): Path to the folder where files will be saved.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename, pdb_str in pdb_dict.items():
        if not filename.endswith(".pdb"):
            filename += ".pdb"
        filepath = os.path.join(output_folder, filename)
        with open(filepath, "w") as f:
            f.write(pdb_str)


def create_data_index(pdb_files: dict[str, str], output_folder: str) -> pd.DataFrame:
    """
    Creates a DataFrame index for PDB files with metadata for downstream processing with ATOMICA.

    Args:
        pdb_files (dict[str, str]): Mapping of filenames to PDB strings.
        output_folder (str): Directory path where PDB files are saved.

    Returns:
        pd.DataFrame: DataFrame with columns including pdb_id, pdb_path, chain identifiers, and ligand information placeholders.
    """

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
    return process_all_pdbs(index_df=pdb_index, dist_th=10.0)


def add_affinities(
    items: list[dict], affinities: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Adds affinity values to items and splits them into training, validation, and test sets.

    Args:
        items (list[dict]): List of data items to which affinities will be added.
        affinities (str): Multiline string where each line contains affinity-related values (expected 5 columns).

    Returns:
        tuple: Three lists of dicts representing (train_data, val_data, test_data) splits with affinity information added.
    """

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

    # Collect all affinity values first for normalization
    affinity_values = []
    for parts in valid_lines:
        col2 = float(parts[1])
        col3 = float(parts[3])
        affinity = col3 - col2
        if affinity <= 0:
            raise ValueError(
                f"Non-positive affinity value ({affinity}) in line: {' '.join(parts)}"
            )
        affinity_values.append(affinity)

    min_aff = min(affinity_values)
    max_aff = max(affinity_values)
    if max_aff == min_aff:
        raise ValueError("All affinity values are the same; cannot normalize.")

    # Assign normalized affinity values to items
    for item, parts, affinity in zip(items, valid_lines, affinity_values):
        normalized_aff = (affinity - min_aff) / (max_aff - min_aff)
        item["affinity"] = {
            "value": affinity,
            "neglog_aff": -np.log10(affinity),
        }

    # Set aside exactly 50 items for test
    test_size = 50
    assert len(items) > test_size, "Not enough items to allocate 15 test samples"

    # First split: remove test set
    remaining_items, test_data = train_test_split(
        items, test_size=test_size, random_state=seed
    )

    # Second split: train and validation from remaining
    train_data, val_data = train_test_split(
        remaining_items, test_size=0.2, random_state=seed  # 20% validation of remaining
    )

    return train_data, val_data, test_data
