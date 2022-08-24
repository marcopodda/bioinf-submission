# A feature-agnostic machine learning framework to improve antigen discovery for bacterial pathogens


## Prepare the environment

First, clone the repository and `cd` into the right directory:

    git clone https://github.com/marcopodda/bioinf-submission.git && cd bioinf-submission

To run the scripts, you need to first setup a conda environment by executing the following command:

    conda env create -f conda/env.yaml

Before running any script, ensure you are in the correct folder (where the repo was cloned) and that you are using the correct environment by executing the following command:

    conda activate bioinf


## Reproducing the experiments

To reproduce the experiments, execute the following script:

    python evaluate.py experiment=<EXPERIMENT> proteome=<PROTEOME> seed=<SEED>

where `<EXPERIMENT>` is either `features` or `pses`, proteome is either the UniProt proteome ID of the species you want to evaluate in LOBO mode (check the `data` folder), or `benchmark` to evaluate the PSE on the iBPA benchmark. `<SEED>` lets you fix a random seed for the experiments.

By default, the experiments will be saved in the folder `experiments`. If you want to reproduce, please rename the folder, otherwise they won't be overwritten to prevent data loss.

The experiments will produce the following files:

- `<SEED>_cv_results.csv`, with the results of the 5-fold cross-validation in .csv format
- `<SEED>_cv.pkl`, a file with the results of the 5-fold cross-validation (more detailed)
- `<SEED>_model.pkl`, the model in pickle format
- `<SEED>_best_params.pkl`, the best hyperparameters found with model selection
- `<SEED>_test.pkl`, the predictions of the model on the LOBO test
- `<SEED>_ranking.pkl`, the proteins ranked by decreasing output probability (for in-vivo prioritization evaluation)
- `<SEED>_config.log`, a log file with experiment configuration.


## Reproducing the tables/plots

The results of the paper are summarized in a Jupyter notebook. Please check  `notebooks/analysis.ipynb`.
