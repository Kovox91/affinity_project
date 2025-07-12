from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_DNA_pdbs,
    create_complex_pdbs,
    save_pdb_files,
    create_data_index,
    process_pdbs,
    add_affinities,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_DNA_pdbs,
                inputs="input_sequences",
                outputs="dna_pdbs",
                name="create_dna_pdbs_node",
            ),
            node(
                func=create_complex_pdbs,
                inputs=dict(
                    mutant_structures="dna_pdbs",
                    complex_template="complex_template_pdb",
                    params="params:complex_pdbs_params"
                ),
                outputs="complex_pdbs",
                name="create_complex_pdbs_node",
            ),
            node(
                func=save_pdb_files,
                inputs=["complex_pdbs", "params:complex_pdb_output_folder"],
                outputs=None,  # no return, side effect only
                name="save_complex_pdb_files_node",
            ),

            node(
                func=create_data_index,
                inputs=["complex_pdbs", "params:complex_pdb_output_folder"],
                outputs="complex_index_df",
                name="create_index_node",
            ),
            node(
                func=process_pdbs,
                inputs=["complex_index_df"],
                outputs="processed_pdbs",
                name="process_pdbs_node",
            ),
            node(
                func=add_affinities,
                inputs=["processed_pdbs", "input_sequences"],
                outputs=["train_items", "test_items", "val_items", "infer_items"],
                name="add_affinities_node",
            ),
        ]
    )
