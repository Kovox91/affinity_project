from Bio.PDB import PDBParser, PDBIO, Superimposer
import sys
import os
from tqdm import tqdm

# Constants
BACKBONE_ATOMS = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]
# Adjust as needed
RESIDUES_COMPLEX = list(range(2, 16))
RESIDUES_MUTANT = list(range(5, 19))

# Utility: Get residues with full backbone atoms
def get_residues_with_full_backbone(structure, chain_id, allowed_residues):
    full_res = []
    for residue in structure[chain_id]:
        if residue.id[1] not in allowed_residues:
            continue
        if all(atom in residue for atom in BACKBONE_ATOMS):
            full_res.append(residue.id[1])
    return full_res

# Utility: Get ordered backbone atoms for selected residues
def get_backbone_atoms_filtered(structure, chain_id, residue_ids):
    atoms = []
    for residue in structure[chain_id]:
        if residue.id[1] not in residue_ids:
            continue
        if all(atom in residue for atom in BACKBONE_ATOMS):
            for atom_name in BACKBONE_ATOMS:
                atoms.append(residue[atom_name])
    return atoms

# Replace chains B and C in complex with mutant A and B
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

# Load structures
parser = PDBParser(QUIET=True)
complex_structure = parser.get_structure("complex", "2ff0.pdb")
complex_structure = complex_structure[0]

# DNA .pdb directory
directory = os.fsencode("../../data/02_intermediate/dna_pdbs")
for file in tqdm(os.listdir(directory), "Creating complexes", unit="file"):
    filename = os.fsdecode(file)
    
    # load mutant structure
    mutant_structure = parser.get_structure("mutant", "../../data/02_intermediate/dna_pdbs/" + filename)
    mutant_structure = mutant_structure[0]

    # Get residues with full backbone
    complex_res_B = set(get_residues_with_full_backbone(complex_structure, "B", RESIDUES_COMPLEX))

    mutant_res_A  = set(get_residues_with_full_backbone(mutant_structure, "A", RESIDUES_MUTANT))

    # Get backbone atoms
    complex_atoms_B = get_backbone_atoms_filtered(complex_structure, "B", RESIDUES_COMPLEX)
    mutant_atoms_A  = get_backbone_atoms_filtered(mutant_structure, "A", RESIDUES_MUTANT)

    if len(complex_atoms_B) != len(mutant_atoms_A):
        sys.exit(f"Backbone atom counts differ: {len(complex_atoms_B)} vs {len(mutant_atoms_A)}")

    # Superimpose
    sup = Superimposer()
    sup.set_atoms(complex_atoms_B, mutant_atoms_A)
    sup.apply(mutant_structure.get_atoms())

    replace_chain(complex_structure, mutant_structure, "B", "A", "B")
    replace_chain(complex_structure, mutant_structure, "C", "B", "C")

    # Save output
    io = PDBIO()
    io.set_structure(complex_structure)
    io.save("../../data/02_intermediate/complexes_pdbs/" + filename)