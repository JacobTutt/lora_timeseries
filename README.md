# M2 Coursework Repository
## LoRA for Time Series Analysis with LLMs

- This repository contains the full pipeline used to preform the evaluation, training, generation and hyperparameter tuning.
- Results were achieved by submitting jobs to a HPC and storing results and metric through .json files and using weights and biases (wandb)


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

Clone the repository from the remote repository (GitLab) to your local machine.

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/m2_coursework/jlt67.git
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
Navigate to the repository’s root directory and install the package dependencies:
```bash
cd jlt67
pip install -r requirements.txt
```

### 4. Set Up a Jupyter Notebook Kernel
To ensure the virtual environment is recognised within Jupyter notebooks, set up a kernel:
```bash
python -m ipykernel install --user --name=env --display-name "Python (M2 Coursework)"
```

### 5. Run the Notebooks
Open the notebooks and select the created kernel **Python (M2 Coursework)** to run the code.


## For Assessment

### Report
Please find the projects report under `Report` directory

### Declaration of Use of Autogeneration Tools
This report made use of Large Language Models (LLMs) to assist in the development of the project.
These tools have been employed for:
- Formatting plots to enhance presentation quality.
- Performing iterative changes to already defined code.
- Debugging code and identifying issues in implementation.
- Helping with Latex formatting for the report.
- Identifying grammar and punctuation inconsistencies within the report.