from pymol import cmd

sequence_path = "seqs.dat"
with open(sequence_path, 'r') as file:
    sequences = file.read().splitlines()

for seq in sequences:
    cmd.fnab(seq, "dna")
    cmd.save("dna_pdbs/" + seq + ".pdb", "dna")
    cmd.delete('dna')


cmd.quit()
