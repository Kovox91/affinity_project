## Outline of the pipeline
1) Run `create_dna.py` and `create_complex_pdbs.py` from the dna_processing folder/step
2) Index the complexes with `indexing.py` from preprocessing This also adds the labels, dummy data for now
3) Process the .pdbs (from the atomica folder):
```
python data/process_pdbs.py \
    --data_index_file ../../../data/02_intermediate/pdb_index.csv \
    --out_path ../../../data/02_intermediate/processed_pdbs.pkl
```
4) add labels to the .pkl file using labelling.py
5) Finetune the model using `ATOMICA/scripts/train_atomica_affinity.sh`


## TODO
Next steps, for easier iterations:
- Put the current workflow into a proper kedro pipeline to automate, especially venv changes.
- Undestand the model
- Improve the fintuning process
- Implement proper data instead of the dummy data.
- Enable everything for remote work (Kubernetes? Docker?)
