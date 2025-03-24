import torch
from accelerate import Accelerator
from .flops_counter import model_evaluation_flops
from torch.utils.data import DataLoader

def evaluate(model, val_dataset, accelerator, lora_ranks, subset = None, print_summary=False):
    """
    Evaluate a model on a validation dataset and estimate FLOP cost.

    This function sets the model to evaluation mode, computes the average 
    cross-entropy loss on the given validation dataset, and estimates the 
    floating point operations (FLOPs) required for evaluation. Optionally, 
    evaluation can be limited to a subset of batches for faster runtime.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to evaluate. Should be wrapped by `accelerator.prepare()`.
    
    val_dataset : torch.utils.data.Dataset
        The validation dataset containing tokenized input sequences.
    
    accelerator : accelerate.Accelerator
        Hugging Face `Accelerator` instance used to handle device placement and model wrapping.

    lora_ranks : int
        The rank of the LoRA decomposition (used for estimating LoRA-specific FLOPs).

    Returns
    -------
    val_loss : float
        The average cross-entropy loss over the validation dataset.

    flop_cost : float
        The estimated number of FLOPs required to evaluate the model over the validation set.
    """


    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    if accelerator is None:
        accelerator = Accelerator()
        model, val_loader = accelerator.prepare(model, val_loader)
    else:
        val_loader = accelerator.prepare(val_loader)
    # Prepare the model for full evaluation - ie Disable dropout, layernorm noise, etc.
    model.eval() 
    # A running total of the validation loss for ech batch which will be averaged at the end
    val_loss = 0.0

    # Estimate evaluation size to automatically calculate FLOPs
    no_tokens = len(val_loader.dataset[0])

    # If subset is valid, limit evaluation to first `subset` batches
    max_batches = len(val_loader)
    if isinstance(subset, int) and subset > 0:
        max_batches = min(subset, len(val_loader))

    # Begin evaluation
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            # Forward pass with teacher forcing (labels = input)
            outputs = model(batch, labels=batch)
            # Accumulate loss
            loss = outputs.loss
            val_loss += loss.item()

    # Estimate FLOP cost based on number of batches evaluated
    flop_cost, _ = model_evaluation_flops(no_tokens, lora_ranks, max_batches, print_summary=print_summary)

    return val_loss / max_batches, flop_cost
