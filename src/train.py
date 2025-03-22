import torch
from accelerate import Accelerator
from .evaluate import evaluate
from .flops_counter import model_training_flops
from tqdm import tqdm

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


def train(model, lora_rank, max_training_steps, batch_size, learning_rate, train_loader, val_loader, early_stopping_patience=3, subset = None, print_summary=False, wandb_run=None):
    """
    Train a LoRA-adapted transformer model using gradient descent, early stopping, and optional Weights & Biases tracking.

    This function performs step-wise training with periodic validation, FLOP tracking, and optional logging to wandb.
    Early stopping is triggered if validation loss doesn't improve after a given number of evaluations.

    Parameters
    ----------
    model : torch.nn.Module
        The transformer model with LoRA modules injected and ready for fine-tuning.

    lora_rank : int
        The rank of the LoRA decomposition, used in FLOP cost estimation.

    max_training_steps : int
        Maximum number of gradient updates (not epochs).

    batch_size : int
        Size of each training mini-batch.

    learning_rate : float
        Learning rate for the Adam optimizer.

    train_loader : torch.utils.data.DataLoader
        DataLoader object for training dataset.

    val_loader : torch.utils.data.DataLoader
        DataLoader object for validation dataset.

    early_stopping_patience : int, optional
        Number of consecutive validation checks with no improvement before early stopping.

    subset : int or None, optional
        If specified, evaluates only the first `subset` batches during validation (useful for speeding up evaluation).

    print_summary : bool, optional
        If True, prints training metrics and FLOP summaries at the end.

    wandb_run : wandb.Run or None, optional
        An active Weights & Biases run for logging metrics. If None, wandb logging is disabled.

    Returns
    -------
    model : torch.nn.Module
        The trained (or early-stopped) model.

    step_tracker : List[int]
        List of training steps at which validation occurred.

    val_loss_tracker : List[float]
        List of validation losses corresponding to `step_tracker`.

    total_flops : float
        Total estimated FLOPs used during training + validation.
    """

    # Optimizer only on trainable (LoRA and LM head) parameters
    optimiser = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)

    # Early stopping controller
    early_stopper = EarlyStopping(patience=early_stopping_patience, mode='min')

    # Use Hugging Face's Accelerator to prepare multi-device training
    accelerator = Accelerator()
    model, optimiser, train_loader, val_loader = accelerator.prepare(model, optimiser, train_loader, val_loader)

    # Assume all sequences are same length → use first item
    token_length = len(train_loader.dataset[0])

    step = 0
    total_eval_cost = 0
    step_tracker = []       # Track steps when validation is run
    val_loss_tracker = []   # Track validation loss per evaluation

    model.train()

    # TQDM progress bar for all training steps
    pbar = tqdm(total=max_training_steps, desc="Training", unit="step")


    while step < max_training_steps:
        for batch in train_loader:
            # Forward and backward pass
            optimiser.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimiser.step()
            step += 1

            # Update progress bar with latest training loss
            pbar.update(1)

            # Validation and logging every 25 steps, if subset is given it will limit the number of batches evaluated on to 
            # the first `subset` batches which are random
            if step % 25 == 0:
                val_loss, flops_cost = evaluate(model, val_loader, accelerator, lora_rank, subset=subset)
                total_eval_cost += flops_cost
                pbar.set_postfix({"train_loss": loss.item(), "val_loss": val_loss})

                step_tracker.append(step)
                val_loss_tracker.append(val_loss)

                # Log metrics to wandb
                if wandb_run is not None:
                    wandb_run.log({
                        "train_loss": loss.item(),
                        "val_loss": val_loss,
                        "step": step,
                        "eval_flops": flops_cost
                    })

                # Early stopping check
                early_stopper(val_loss)
                if early_stopper.early_stop:
                    print(
                        f"Early stopping occurred at step {step} "
                        f"(val_loss = {val_loss:.4f}, patience = {early_stopper.patience})"
                    )
                    if wandb_run is not None:
                        wandb_run.log({"early_stopping_step": step})
                    break

            if step >= max_training_steps:
                break

    pbar.close()

    # Final full validation after training ends
    val_loss_final, flops_cost = evaluate(model, val_loader, accelerator, lora_rank)
    total_eval_cost += flops_cost
    step_tracker.append(step)
    val_loss_tracker.append(val_loss_final)

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

    return model, step_tracker, val_loss_tracker, total_flops