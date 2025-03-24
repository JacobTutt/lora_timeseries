from .preprocessor import preprocessor
from .full_model import full_model
from .train import train
from .dataset import TimeSeriesData
import wandb

# ------------------------ Training Function ------------------------ #
def hyperparam_wandb(config=None):
    """
    Entry point for hyperparameter tuning with Weights & Biases sweeps.

    This function is designed to be used with `wandb.agent()` for running individual training jobs
    during a sweep. It loads a LoRA-injected Qwen model, prepares a tokenized dataset, builds 
    PyTorch DataLoaders, and launches the training process using the `train()` function.

    The function expects a `wandb.config` object containing all necessary hyperparameters, including:
    - `learn_rate` : float
        The learning rate for the Adam optimizer.
    - `lora_rank` : int
        Rank of the Low-Rank Adaptation (LoRA) decomposition.
    - `token_length` : int
        Token sequence length used for chunking time series data.
    - `max_steps` : int
        Maximum number of training steps.
    - `train_sequences` : int
        Number of time series sequences to use for training.
    - `batch_size` : int
        Number of token sequences per training batch.
    - `decimal_places` : int
        Decimal precision used for rounding in the preprocessed dataset.
    - `subset` : int or None
        If set, limits number of validation batches used during mid-training evaluation (useful to speed up sweeps).

    Workflow:
    ---------
    - Loads the model via `full_model(lora_rank=...)`
    - Prepares dataset with `preprocessor(...)`
    - Constructs `TimeSeriesData` and `DataLoader`s
    - Calls `train()` with all required parameters and `wandb_run=wandb` to enable metric logging

    Returns
    -------
    None

    Notes
    -----
    This function is intended to be passed to `wandb.agent(...)` and does not return a trained model.
    Final metrics like validation loss, FLOP counts, and step counts are logged to the W&B dashboard.
    """
    with wandb.init(config=config):
        config = wandb.config

        learning_rate = config.learn_rate # learning rate
        lora_rank = config.lora_rank # LoRA rank to give model
        token_length = config.token_length # token length
        max_training_steps = config.max_steps # maximum training steps
        batch_size = config.batch_size # batch size
        decimal_places = config.decimal_places # decimal places used for data
        subset = config.subset # subset of data to evaluate
        eval_freq = config.eval_freq # evaluate every 10 steps

        # Load model and tokeniser
        model, tokeniser, device = full_model(lora_rank=lora_rank)
        
        # Load and tokenise dataset
        train_set_total, val_set_total, test_set_total = preprocessor('lotka_volterra_data.h5', percentile=90, decimal_places=decimal_places, train_fraction=0.7, validation_fraction=0.15, shuffle=False, print_summary=False)

        train_dataset = TimeSeriesData(train_set_total, tokeniser, max_length=token_length, stride=token_length/2)
        val_dataset = TimeSeriesData(val_set_total, tokeniser, max_length=token_length, stride=token_length)

        # ------------------------ Call Your Training Loop ------------------------ #

        _, _, _ = train(model=model, lora_rank=lora_rank, max_training_steps=max_training_steps, batch_size=batch_size, learning_rate=learning_rate,
            no_tokens = token_length, train_dataset=train_dataset, val_dataset=val_dataset, early_stopping_patience=5,
            subset=subset, # Evaluate only om  a certain no batches during validation in the middle of training this is useful for speeding up evaluation
            eval_freq = eval_freq, # Evaluate on validation every 10 steps
            print_summary=False,
            wandb_run=wandb  # Pass wandb for logging
        )

