# DigitalGuardRails

This is the repository for AngelEye Health's DigitalGuardRails project.


## Installation

Please first [install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (anaconda or miniconda) to get started.

Run the following code to construct the conda environment.
	
	conda env create -f environment.yml

## Run As...

### Jupyter Notebook

Please navigate to and edit [DigitalGuardRails.ipynb](https://github.com/angelptins/DigitalGuardRails/blob/main/DigitalGuardRails.ipynb) to interact with the Digital Guard Rails pipeline. 

The following code opens an interactive session from the current working directory.

	jupyter notebook

Sample input, output, and animation files from the DGR pipeline are available in this repository.

### Python Script

Run the following code as an executable Python script with the flags.

	python DigitalGuardRails.py \
		--fname-excel ./test_input.xlsx \
		--fname-csv ./test_output.csv \
		--fname-ani test_animation.mp4

More information on how to use the script can be seen by running:

	python DigitalGuardRails.py --help






