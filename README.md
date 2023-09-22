# DispaRisk
This repository contains the code for the experiments of the paper DispaRisk: A Framework for Assessing and Interpreting Disparity Risks in Datasets.

To run the experiments, please ensure you create a Python environment with python>=3.8.6.

### CelebA Dataset
The folder ‘celeba’ contains all the code and experiments on the CelebA dataset. First, you need to download the celeba.hdf5 file from [this link](https://drive.google.com/drive/folders/1lbw4laF9vsNKVAzTZoCjla-dGNAAFVtz?usp=sharing) and provide the path to this file to the `--data-dir`.

### Tabular Dataset
The folder `tabular_datasets` contains all the experiments for the three tabular datasets: Census Income (CI), Dutch Census (DC), and Compas Recidivism (CR). You need to provide the name of the dataset for the --datasets parameter. You also need to provide the corresponding model: (1) sgd_lr for Linear family, (2) mlp_one_layer for 1MLP, and (3) two_one_layer for 2MLP.

In each folder, you will find Jupyter notebooks with all the analysis not reported in the paper. Please check `Analysis_V_Information.ipynb`, `sampling_tests.ipynb`, and `plots.ipynb` files in the `tabular_datasets` folder.


Checkpoints and Results
Resulting models trained on the Tabular and CelebA datasets can be found in this [Google Drive](https://drive.google.com/drive/folders/1yZOlWHpq2s83-CXBcVkEdLVdhw7wLtZo?usp=drive_link) folder. You will also find the raw data for computing the results reported in the paper: PVIs, fairness metrics, and performances.