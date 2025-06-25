## Setup
To run this, Linux is required.
Run the following:  
`conda env create -n affinity_pipeline -f environment_full.yaml`  
This is going to create the affinity_pipeline environment. Activate it:  
`conda activate affinity_pipeline`  
TBC


## TODO
Next steps, for easier iterations:
- Put the current workflow into a proper kedro pipeline to automate, especially venv changes.
- Undestand the model
- Improve the fintuning process
- Implement proper data instead of the dummy data.
- Enable everything for remote work (Kubernetes? Docker?)
