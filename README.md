# [LoRA for Time Series Analysis with LLMs](https://lora-timeseries.readthedocs.io/en/latest/)

- This repository contains the full pipeline used to preform the evaluation, training, generation and hyperparameter tuning.
- Results were achieved by submitting jobs to a HPC and storing results and metric through .json files and using weights and biases (wandb)
- Documentation is avaliable: [here](https://lora-timeseries.readthedocs.io/en/latest/)

## Pipleine (`src`)
### `Preprocessor` 
— Scaling & Encoding Time Series from HDF5
### `Decoder` 
— Convert Encoded String Back to Time Series Arrays
### `TimeSeriesData` 
— Tokenised Chunked Dataset for Transformers
### `Full_model` 
— Load and LoRA-Inject a Pretrained Transformer
### `EarlyStopping` 
— Prevent Overfitting with Adaptive Termination
### `Train` 
— LoRA Transformer Training with Validation, Logging, and Early Stopping
### `Evaluate` 
— Validation Loss and FLOPs Estimation for LoRA Models
### `Generate` 
— Autoregressive Sequence Prediction from Context Using LoRA Model
### `Hyperparameter_run` 
— Hyperparameter Training Run with LoRA
### `Hyperparam_wandb` 
— Run Sweep Trial with LoRA and W&B Integration
### `Flops Calculations` 
- The underlying flops calcs later built it to all relavent functions above
### `Analysis` 
- Additional Evaluation Metrics (MSE/MAE/MAPE) and plotting

## Notebooks
- Provides a detailed discussion and breakdown of approach takens
- Provides analysis and plotting of results from .json file results

---

## Installation Instructions

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

## Report
A report for this project can be found under the Report directory of the repository
