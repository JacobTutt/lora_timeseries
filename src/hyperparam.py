from .preprocessor import preprocessor
from .full_model import full_model
from .train import train
from .dataset import TimeSeriesData
import wandb
import joblib

def hyperparameter_run(data_path, learning_rate = 1e-4, lora_rank = 2, token_length = 512, max_training_steps = 100, batch_size = 2, decimal_places = 3 ,subset = 2, eval_freq = 10, patience = 5, wandb_project = None, save_path = None):
    """
    Perform a single training run with specified hyperparameters using a LoRA-adapted transformer model.

    This function loads a transformer model with LoRA, prepares the tokenized time series dataset,
    and trains the model using the specified hyperparameters. The training process includes
    early stopping, FLOP tracking, and optional logging to Weights & Biases (wandb).

    Parameters
    ----------
    data_path : str
        Path to the time series data
    learning_rate : float, optional
        Learning rate for the Adam optimizer (default: 1e-4).
    lora_rank : int, optional
        Rank used for LoRA decomposition (default: 2).
    token_length : int, optional
        Length of each tokenized input sequence (default: 512).
    max_training_steps : int, optional
        Maximum number of training steps (default: 100).
    batch_size : int, optional
        Size of each training mini-batch (default: 2).
    decimal_places : int, optional
        Number of decimal places to round the data for tokenization (default: 3).
    subset : int, optional
        Number of batches to use during evaluation subset (default: 2).
    eval_freq : int, optional
        Frequency (in steps) at which validation is performed (default: 10).
    patience : int, optional
        Number of evaluation steps to wait for improvement before early stopping (default: 5).
    wandb_project : str, optional
        Name of the Weights & Biases project for logging (default: None, no wandb logging).
    save_file : str, optional
        Path to save the results dictionary using joblib (default: None, no saving).

    Returns
    -------
    val_step_tracker : List[int]
        Steps at which validation was performed.
    train_step_tracker : List[int]
        Steps at which training loss was recorded.
    val_loss_tracker : List[float]
        Validation loss recorded at each evaluation step.
    train_loss_tracker : List[float]
        Training loss recorded at each step.
    total_flops : float
        Total number of FLOPs used for training and evaluation.

    Notes
    -----
    - If `wandb_project` is provided, ensures you are logged in to wandb before calling.
    - If `save_file` is provided, the results are saved as a joblib file.
    """
    # Load model and tokeniser
    model, tokeniser, device = full_model(lora_rank=lora_rank)

    # Load and tokenise dataset
    train_set_total, val_set_total, test_set_total = preprocessor(data_path, percentile=90, decimal_places=decimal_places, train_fraction=0.7, validation_fraction=0.15, shuffle=False, print_summary=False)

    # Initialise wandb run if wandb is passed
    if wandb_project is not None:
        wandb_run = wandb.init(project=wandb_project, name=f"LR{learning_rate}LR{lora_rank}TL{token_length}")
    else:
        wandb_run = None

    train_dataset = TimeSeriesData(train_set_total, tokeniser, max_length=token_length, stride=token_length/2)
    val_dataset = TimeSeriesData(val_set_total, tokeniser, max_length=token_length, stride=token_length)


    # ------------------------ Call Your Training Loop ------------------------ #

    _, val_step_tracker, train_step_tracker, val_loss_tracker, train_loss_tracker, val_loss_final, training_flops, total_eval_cost = train(model=model, lora_rank=lora_rank, max_training_steps=max_training_steps, batch_size=batch_size, no_tokens = token_length ,learning_rate=learning_rate, train_dataset=train_dataset, val_dataset=val_dataset, early_stopping_patience = patience,
        subset=subset, # Evaluate only om  a certain no batches during validation in the middle of training this is useful for speeding up evaluation
        eval_freq = eval_freq, # Evaluate on validation every 10 steps
        print_summary=False,
        wandb_run=wandb_run,  # Pass wandb for logging
        save_path=save_path) # Save the model to the path

    total_flops = training_flops + total_eval_cost


    return val_step_tracker, train_step_tracker, val_loss_tracker, train_loss_tracker, total_flops


