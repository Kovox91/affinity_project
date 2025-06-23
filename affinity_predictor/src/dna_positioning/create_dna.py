from pathlib import Path
from pymol import cmd
from tqdm import tqdm

# Define input and output paths
input_path = Path("../../data/01_raw/seqs.dat")
output_dir = Path("../../data/02_intermediate/dna_pdbs")
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# Load sequences
with input_path.open("r") as file:
    sequences = file.read().splitlines()

# Process each sequence
for seq in tqdm(sequences, desc="Generating PDBs"):
    output_file = output_dir / f"{seq}.pdb"
    
    if output_file.exists():
        continue  # Skip if file already exists
    
    cmd.fnab(seq, "dna")
    cmd.save(str(output_file), "dna")
    cmd.delete("dna")

cmd.quit()
