## Outline of the pipeline
1) Run `create_dna.py` and `create_complex_pdbs.py` from the dna_processing folder/step
2) Index the complexes with `indexing.py` from preprocessing #TODO ADD LABELS!
3) Process the .pdbs (from the atomica folder):
```
python data/process_pdbs.py \
    --data_index_file ../../../data/02_intermediate/pdb_index.csv \
    --out_path ../../../data/02_intermediate/processed_pdbs.pkl
```
4) Add the affinity values to the .pkl file using `labelling.py` in preprocessing (Currently this add dummy data!)
5) Finetune the model using `ATOMICA/scripts/train_atomica_affinity.sh`