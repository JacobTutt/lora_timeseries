# **LoRA for Time Series Analysis with LLMs**

## Author: Jacob Tutt, Department of Physics, University of Cambridge

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/lora-timeseries/badge/?version=latest)](https://lora-timeseries.readthedocs.io/en/latest/?badge=latest)

## Description
- This repository contains the full pipeline used to preform the evaluation, training, generation and hyperparameter tuning.
- Results were achieved by submitting jobs to a HPC and storing results and metric through .json files and using weights and biases (wandb)
- Documentation is avaliable: [here](https://lora-timeseries.readthedocs.io/en/latest/)

This repository forms part of the submission for the MPhil in Data Intensive Science's M2 Deep Learning Course at the University of Cambridge.

## Table of Contents
- [Pipeline Functionalities](#pipelines)
- [Notebooks](#notebooks)
- [Documentation](#documentation)
- [Installation](#installation-and-usage)
- [License](#license)
- [Support](#support)
- [Author](#author)

## Pipelines
This works presents moduluar functions allowing easy tockenisation, training, hyperparameter tuning and evaluation.

| Functions | Description |
|----------|-------------|
| [Preprocessor](src/preprocessor.py) | Scaling & Encoding Time Series from HDF5 |
| [Decoder](src/decoder.py) | Convert Encoded String Back to Time Series Arrays |
| [TimeSeriesData](src/dataset.py) | Tokenised Chunked Dataset for Transformers |
| [Full_model](src/full_model.py) | Load and LoRA-Inject a Pretrained Transformer |
| [EarlyStopping](src) | Prevent Overfitting with Adaptive Termination |
| [Train](src/train.py) | LoRA Transformer Training with Validation, Logging, and Early Stopping |
| [Evaluate](src/evaluate.py) | Validation Loss and FLOPs Estimation for LoRA Models |
| [Generate](src/generate.py) | Autoregressive Sequence Prediction from Context Using LoRA Model |
| [Hyperparameter_run](src) | Hyperparameter Training Run with LoRA |
| [Hyperparam_wandb](src) | Run Sweep Trial with LoRA and W&B Integration |
| [Flops Calculations](src/flops_counter.py) | The underlying flops calcs later built it to all relavent functions above |
| [Analysis](src/analysis.py) | Additional Evaluation Metrics (MSE/MAE/MAPE) and plotting |


## Notebooks
- Provides a detailed discussion and breakdown of approach takens
- Provides analysis and plotting of results from .json file results

## Installation and Usage

To run the notebooks, please follow these steps:

### 1. Clone the Repository

Clone the repository from the remote repository (GitHub) to your local machine.

```bash
git clone https://github.com/JacobTutt/lora_timeseries.git
cd jlt67
```

### 2. Create a Fresh Virtual Environment
Use a clean virtual environment to avoid dependency conflicts.
```bash
python -m venv env
source env/bin/activate   # For macOS/Linux
env\Scripts\activate      # For Windows
```

### 3. Install the dependencies
Navigate to the repository’s root directory and install the pipeline and its dependencies:
```bash
cd jlt67
pip install -e .
```

### 4. Set Up a Jupyter Notebook Kernel
To ensure the virtual environment is recognised within Jupyter notebooks, set up a kernel:
```bash
python -m ipykernel install --user --name=env --display-name "Python (Lora)"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel **Python (Lora)** to run the code.

## For Assessment
- The associated project report can be found under [Project Report](report/main.pdf). 

## License
This project is licensed under the [MIT License](https://opensource.org/license/mit/) - see the [LICENSE](LICENSE.txt) file for details.

## Support
If you have any questions, run into issues, or just want to discuss the project, feel free to:
- Open an issue on the [GitHub Issues](https://github.com/JacobTutt/stat_frequentist_analysis/issues) page.  
- Reach out to me directly via [email](mailto:jacobtutt@icloud.com).

## Author
This project is maintained by Jacob Tutt 
