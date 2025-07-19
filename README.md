# **LoRA for Time Series Analysis with LLMs**

## Author: Jacob Tutt, Department of Physics, University of Cambridge

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/lora-timeseries/badge/?version=latest)](https://lora-timeseries.readthedocs.io/en/latest/?badge=latest)

## Description
This repository explores the use of Large Language Models (LLMs) for time series forecasting, leveraging LoRA-injected Qwen transformers to model coupled nonlinear dynamics. Under a strict compute budget of $10^{17}$ FLOPs, the pipeline demonstrates efficient downstream training and evaluation on the Lotka–Volterra (Predator–Prey) system. The work combines modular code, automated hyperparameter tuning, and quantitative analysis to investigate both the representational capacity and efficiency of tuned-adapted LLMs in forecasting physical time-series systems.

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
This works presents moduluar functions allowing easy tokenisation, training, hyperparameter tuning and evaluation. Which can be briefly broken down as such:

| Functions | Description |
|----------|-------------|
| [Preprocessor](src/preprocessor.py) | Loads and scales predator–prey time series data, encodes it into string format suitable for tokenisation, and splits it into train/validation/test sets. |
| [Decoder](src/decoder.py) | Converts an encoded string back into arrays prey and predator populations (reversing the preprocessing step). |
| [TimeSeriesData](src/dataset.py) | PyTorch-compatible Dataset which includes the tokeniser and converts pre-encoded time-series strings into fixed-length, stride-chunked token sequences for efficient training and evaluation. |
| [Full_model](src/full_model.py) | Loads a pretrained Qwen Transformer and optionally injects LoRA layers (rank-configurable) into its query and value projections for efficient fine-tuning. |
| [Train](src/train.py) | Performs full fine-tuning of a LoRA-injected Transformer using mini-batches, validation loss tracking, early stopping, FLOP estimation, and optional Weights & Biases logging. |
| [Evaluate](src/evaluate.py) | Computes the average validation loss and estimates the FLOPs required for evaluation.|
| [Generate](src/generate.py) | Autoregressively completes multivariate time series using a trained LoRA Transformer model. |
| [Flops Calculations](src/flops_counter.py) | The underlying flops calculations which are incorporated into overall functions to allow estimation of compute required for each proccess.
| [Analysis](src/analysis.py) | Tools for quantitative and visual evaluation of model predictions. Includes error metric plots (MAE/MSE/MAPE), performance comparisons (trained vs. untrained), FLOP-based efficiency summaries, and hyperparameter tuning diagnostics. |
| [Full run script](Full_run_script.py) | End-to-end execution pipeline for training, validating, and testing a LoRA-adapted transformer. Handles preprocessing, model setup, early stopping, FLOP tracking, W&B logging, and final evaluation. |

Additionally, we present a configurable pipelines ([sweep](Full_hyper_wandbsweep.py), [yaml](wandb_hyperparam.yaml)) for automated hyperparameter tuning with Weights & Biases, enabling systematic exploration of parameters such as learning rate, LoRA rank, token length, and data precision through flexible sweep configurations.

## Notebooks

The [notebooks](notebooks) in this repository serve as walkthroughs for the analysis performed. They include derivations of the mathematical implementations, explanations of key choices made and breakdown of approach takens. Although the results were achieved by submitting jobs to a HPC and storing results and metric through .json files and using weights and biases (wandb).

| Notebooks | Description |
|----------|-------------|
| [Preprocessing](notebooks/1_Preproccessing_Investigation.ipynb) | Provides a discussion of the different options for scaling methods and the string configurations and the reasoning behind what was implemented. |
| [Flops Calculations](notebooks/2_Flops_Calc.ipynb) | Details the Qwen + LoRA model architecture, outlines the mathematical formulation of FLOP computations, and provides example calculations using the implemented function. |
| [Flops Budget Plan](notebooks/3_Flops_Plan.ipynb) | Establishes a pre-experiment FLOP budget estimate and descibes how the FLOP tracking is dynamically computed and stored using JSON-based logging across training and evaluation phases. |
| [Initial Performance](notebooks/4_Initial_Preformance.ipynb) | Compares untrained Qwen model performance using two complementary evaluation strategies: cross-entropy loss on token prediction and numerical error metrics (MSE, MAE, MAPE) via autoregressive generation. |
| [Default Lora Model](notebooks/5_Lora_Model.ipynb) | Validates the training pipeline by overfitting a LoRA-injected Qwen model on a minimal dataset and benchmarking performance on the full dataset using default hyperparameters for comparisons with subsequent hyperparameter tuning. |
| [Hyperparameter Tuning](notebooks/6_HyperParam_Tune.ipynb) | Systematically explores key tuning parameters - decimal precision, learning rate, LoRA rank, and context length — via single-axis sweeps under a strict FLOP budget, identifying optimal settings for full training. |
| [Extending Training Run](notebooks/7_Final_Training.ipynb) | Executes final training with optimal hyperparameters, comparing downstream prediction metrics (MAE, MSE, MAPE) across multiple timesteps highlighting both short-term accuracy and long-horizon generalisation. |
| [Final Flops Calculation](notebooks/8_Final_Flops_Count.ipynb) | A final outline of the total Flops used throughout the work simulating real life compute constraints. |

## Documentation
Detailed documentation of the available pipelines is available [here](https://lora-timeseries.readthedocs.io/en/latest/).

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
- Open an issue on the [GitHub Issues](https://github.com/JacobTutt/lora_timeseries/issues) page.  
- Reach out to me directly via [email](mailto:jacobtutt@icloud.com).

## Author
This project is maintained by Jacob Tutt 
