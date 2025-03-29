from src import preprocessor
from src import full_model
from src import train
from src import evaluate
from src import TimeSeriesData
import wandb

# This function preforms everythign you need overall:

def full_run(data_path, learning_rate = 1e-4, lora_rank = 2, token_length = 512, max_training_steps = 100, batch_size = 2, decimal_places = 3 ,subset = 2, eval_freq = 10, patience = 5, wandb_project = None, save_path = None):
    """
    Run a complete training, validation, and test evaluation pipeline for a LoRA-adapted transformer model.

    This function loads a LoRA-enhanced transformer model, processes a time-series dataset,
    performs training with early stopping and FLOP tracking, and finally evaluates the model
    on a held-out test set. It supports optional Weights & Biases logging and result saving.

    Parameters
    ----------
    data_path : str
        Path to the raw time-series data file.
    learning_rate : float, optional (default=1e-4)
        Learning rate for the Adam optimizer.
    lora_rank : int, optional (default=2)
        LoRA decomposition rank applied to model attention layers.
    token_length : int, optional (default=512)
        Maximum token sequence length for model input.
    max_training_steps : int, optional (default=100)
        Maximum number of training steps.
    batch_size : int, optional (default=2)
        Number of samples per training batch.
    decimal_places : int, optional (default=3)
        Precision used when rounding data during tokenization.
    subset : int, optional (default=2)
        Number of batches to use during intermediate validation.
    eval_freq : int, optional (default=10)
        Frequency (in steps) at which validation is performed during training.
    patience : int, optional (default=5)
        Early stopping patience; training halts if no improvement for this many validations.
    wandb_project : str or None, optional
        If provided, logs training metrics to the specified Weights & Biases project.
    save_path : str or None, optional
        If provided, path to save the trained model weights and evaluation results.

    Returns
    -------
    val_step_tracker : List[int]
        List of steps where validation was performed.
    train_step_tracker : List[int]
        List of steps where training loss was recorded.
    val_loss_tracker : List[float]
        Validation loss at each evaluation step.
    train_loss_tracker : List[float]
        Training loss at each training step.
    val_loss_final : float
        Final validation loss after training completion or early stopping.
    test_loss : float
        Cross-entropy loss on the held-out test dataset.
    total_flops : float
        Total estimated FLOPs consumed across training, validation, and testing.

    Notes
    -----
    - Requires the `generate`, `TimeSeriesData`, `train`, and `evaluate` functions to be properly implemented.
    - If `wandb_project` is provided, make sure you are logged into Weights & Biases.
    """
    # Load model and tokeniser
    model, tokeniser, device = full_model(lora_rank=lora_rank)

    # Load and tokenise dataset
    train_set_total, val_set_total, test_set_total = preprocessor(data_path, percentile=90, decimal_places=decimal_places, train_fraction=0.7, validation_fraction=0.15, shuffle=True, print_summary=False)

    # Initialise wandb run if wandb is passed
    if wandb_project is not None:
        wandb_run = wandb.init(project=wandb_project, name=f"LR{learning_rate}LR{lora_rank}TL{token_length}")
    else:
        wandb_run = None

    train_dataset = TimeSeriesData(train_set_total, tokeniser, max_length=token_length, stride=token_length/2)
    val_dataset = TimeSeriesData(val_set_total, tokeniser, max_length=token_length, stride=token_length)
    test_dataset = TimeSeriesData(test_set_total, tokeniser, max_length=token_length, stride=token_length)

    # ------------------------ Call Your Training Loop ------------------------ #

    model, val_step_tracker, train_step_tracker, val_loss_tracker, train_loss_tracker, val_loss_final, training_flops, total_eval_cost = train(model=model, lora_rank=lora_rank, max_training_steps=max_training_steps, batch_size=batch_size, no_tokens = token_length ,learning_rate=learning_rate, train_dataset=train_dataset, val_dataset=val_dataset, early_stopping_patience = patience,
        subset=subset, # Evaluate only om  a certain no batches during validation in the middle of training this is useful for speeding up evaluation
        eval_freq = eval_freq, # Evaluate on validation every 10 steps
        print_summary=False,
        wandb_run=wandb_run,  # Pass wandb for logging
        save_path=save_path) # Save the model to the path


    # ------------------------ Evaluate on Test Set ------------------------ #

    test_loss, test_flop_cost = evaluate(model, test_dataset, accelerator = None, lora_ranks= lora_rank, subset = None, print_summary=True)

    total_flops = training_flops + total_eval_cost + test_flop_cost

    return val_step_tracker, train_step_tracker, val_loss_tracker, train_loss_tracker,val_loss_final, test_loss, total_flops


