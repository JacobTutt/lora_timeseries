import torch
from accelerate import Accelerator
from .evaluate import evaluate
from .flops_counter import model_training_flops
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
# ------------------------ EarlyStopping ------------------------ #
class EarlyStopping:
    """
    Implements early stopping to terminate training when validation performance stops improving.

    This class monitors a score (e.g., validation loss or accuracy) and stops training 
    if it fails to improve after a specified number of evaluation steps (patience).

    It supports both minimization (e.g., for loss) and maximization (e.g., for accuracy or BLEU)
    depending on the selected mode.

    Parameters
    ----------
    patience : int, optional (default=5)
        Number of consecutive evaluations without improvement before stopping.

    mode : str, optional (default='min')
        Whether to consider lower values as better ('min') or higher values as better ('max').
    """

    def __init__(self, patience=5, mode='min'):

        self.patience = patience                  # Number of steps to wait before stopping
        self.counter = 0                          # Counter for consecutive bad epochs
        self.best_score = None                    # Best score seen so far
        self.early_stop = False                   # Whether early stopping condition is met
        self.mode = mode                          # Optimization mode: 'min' or 'max'

    def __call__(self, current_score):
        """
        Evaluate the current score and update early stopping status.

        Parameters
        ----------
        current_score : float
            The current evaluation metric to compare with the best score.
        """

        # If this is the first score seen it is automatically the best
        if self.best_score is None:
            self.best_score = current_score

        # If there is no improvement, increment the counter
        elif ((self.mode == 'min' and current_score >= self.best_score) or
            (self.mode == 'max' and current_score <= self.best_score)):

            self.counter += 1  

            # If the counter exceeds the patience, stop training
            if self.counter >= self.patience:
                self.early_stop = True 

        # If improvement found — reset counter and update best score
        else:
            self.best_score = current_score
            self.counter = 0


def train(model, lora_rank, max_training_steps, batch_size, no_tokens, learning_rate, train_dataset, val_dataset, early_stopping_patience=5, subset = None, eval_freq =10, print_summary=False, wandb_run=None, save_path = None):
    """
    Trains a LoRA-adapted transformer model with early stopping, validation, FLOP tracking,
    and optional Weights & Biases (wandb) logging.

    The training loop supports gradient descent with periodic validation, early stopping based
    on validation loss, and efficient FLOP estimation. Evaluation is optionally performed on a
    subset of the validation data for speed. Progress is shown with a live progress bar.

    Parameters
    ----------
    model : torch.nn.Module
        The transformer model with LoRA adapters injected and ready for fine-tuning.

    lora_rank : int
        The rank used for LoRA decomposition; also used in FLOP estimation.

    max_training_steps : int
        The total number of gradient update steps to perform.

    batch_size : int
        Number of sequences per training batch.

    no_tokens : int
        Number of tokens per sequence (used for FLOP cost estimation).

    learning_rate : float
        The learning rate used by the Adam optimizer.

    train_dataset : torch.utils.data.Dataset
        The dataset of input sequences for training.

    val_dataset : torch.utils.data.Dataset
        The dataset used for validation during training and final evaluation.

    early_stopping_patience : int, optional (default=5)
        Number of consecutive validation checks with no improvement before stopping training.

    subset : int or None, optional (default=None)
        If provided, limits validation during training to the first `subset` batches.

    eval_freq : int, optional (default=10)
        How often (in steps) to run validation and logging.

    print_summary : bool, optional (default=False)
        If True, prints a summary of training duration and compute cost.

    wandb_run : wandb.Run or None, optional
        An active Weights & Biases run object. If provided, logs all metrics to wandb.

    Returns
    -------
    model : torch.nn.Module
        The trained model after completion or early stopping.

    val_step_tracker : List[int]
        List of steps at which validation was performed.

    train_step_tracker : List[int]
        List of steps at which training losses were recorded.

    val_loss_tracker : List[float]
        Validation loss values corresponding to `val_step_tracker`.

    train_loss_tracker : List[float]
        Training loss values recorded per evaluation step.

    val_loss_final : float
        Final validation loss computed after the last training step.

    training_flops : float
        Estimated number of floating-point operations (FLOPs) used for training.

    total_eval_cost : float
        Estimated number of FLOPs used for all validation evaluations
    
    save_path : str or None
        If provided, saves a json file with training results to the specified path    
    
    """

    # The data loader for training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # The dataloader for the validation is within the evaluate function to speed up the process

    # Optimiser only on trainable (LoRA and LM head) parameters
    optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)

    # Early stopping controller
    early_stopper = EarlyStopping(patience=early_stopping_patience, mode='min')

    # Log hyperparameters to wandb
    if wandb_run is not None:
        wandb_run.log({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "no_tokens": no_tokens,
        })

    # Use Hugging Face's Accelerator to prepare multi-device training
    accelerator = Accelerator()
    model, optimiser, train_loader = accelerator.prepare(model, optimiser, train_loader)

    # as all all sequences are same length use first item
    token_length = len(train_loader.dataset[0])

    step = 0
    total_eval_cost = 0
    train_step_tracker = []       # Track steps when validation is run
    val_step_tracker = []         # Track steps when validation is run
    val_loss_tracker = []   # Track validation loss per evaluation
    train_loss_tracker = [] # Track training loss per evaluation

    model.train()

    # progress bar
    pbar = tqdm(total=max_training_steps, desc="Training", unit="step")

    # Early stopping flag
    stop_training = False
    early_stopping_st = None
    # iterate over training steps as long as early stopping condition is not met
    while step < max_training_steps and not stop_training:
        for batch in train_loader:

            # Forward and backward pass
            optimiser.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimiser.step()
            step += 1

            # Update progress bar
            pbar.update(1)
            train_step_tracker.append(step)
            train_loss_tracker.append(loss.item())

            # Validation and logging every 25 steps, if subset is given it will limit the number of batches evaluated on to 
            # the first `subset` batches which are random
            if (step % eval_freq == 0) or (step == 1):
                val_loss, flops_cost = evaluate(model, val_dataset, accelerator, lora_rank, subset=subset)
                # Convert back to training mode
                model.train()
                total_eval_cost += flops_cost
                pbar.set_postfix({"train_loss": loss.item(), "val_loss": val_loss})

                val_step_tracker.append(step)
                val_loss_tracker.append(val_loss)

                # Log metrics to wandb
                if wandb_run is not None:
                    wandb_run.log({
                        "train_loss": loss.item(),
                        "val_loss": val_loss,
                        "step": step,
                        "eval_flops": total_eval_cost
                    })

                # Early stopping check which will break the loop if the early stopping condition is met
                early_stopper(val_loss)
                if early_stopper.early_stop:
                    # Log the early stopping step to wandb
                    if wandb_run is not None:
                        stop_training = True
                        print(f"Early stopping at step {step}, validation loss: {val_loss}")
                        early_stopping_st = step
                        wandb_run.log({"early_stopping_step": step})
                    # Break the loop
                    break
            else:
                # This ensure we log the training loss even if we don't evaluate to wandb

                if wandb_run is not None:
                    wandb_run.log({
                        "train_loss": loss.item(),
                        "step": step,
                    })

            if step >= max_training_steps:
                break

    pbar.close()

    # Final full validation after training ends
    val_loss_final, flops_cost = evaluate(model, val_dataset, accelerator, lora_rank, subset = None, print_summary=False)
    total_eval_cost += flops_cost


    # Compute training FLOPs
    training_flops, _ = model_training_flops(
        no_tokens=token_length,
        lora_ranks=lora_rank,
        batch_size=batch_size,
        num_steps_training=step,
        print_summary=print_summary
    )

    # Total compute
    total_flops = training_flops + total_eval_cost

    # Final logging
    if wandb_run is not None:
        wandb_run.log({
            "final_val_loss": val_loss_final,
            "actual_training_steps": step,
            "training_flops": training_flops,
            "total_eval_flops": total_eval_cost,
            "total_flops": total_flops
        })

    # Optional CLI summary
    if print_summary:
        print(f"Actual training steps preformed: {step}")
        print(f"Final validation loss: {val_loss_final}")
        print(f"Total training FLOPs: {training_flops:.3e}")
        print(f"Total evaluation FLOPs: {total_eval_cost:.3e}")


    if save_path is not None:
        # Save to joblib file with dictionary of all the data
        results_run = {
            "learning_rate": float(learning_rate),
            "batch_size": float(batch_size),
            "no_tokens": float(no_tokens),
            "val_step_tracker": list(val_step_tracker),
            "train_step_tracker": list(train_step_tracker),
            "val_loss_tracker": list(val_loss_tracker),
            "train_loss_tracker": list(train_loss_tracker),
            "val_loss_final": float(val_loss_final),
            "training_flops": float(training_flops),
            "total_eval_cost": float(total_eval_cost),
            "early_stopping_step": early_stopping_st,
            "stopping_reason": "early" if early_stopping_st is not None else "max_steps"
        }


        with open(save_path, "w") as f:
            json.dump(results_run, f, indent=4)

    return model, val_step_tracker, train_step_tracker, val_loss_tracker, train_loss_tracker, val_loss_final, training_flops, total_eval_cost