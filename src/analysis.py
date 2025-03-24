import numpy as np
from .generate import generate
import matplotlib.pyplot as plt


def model_analysis_mse(model, no_random_samples, tokeniser, dataset, inference_split=0.8, randomness=False):
    """
    Evaluate a model on a validation dataset and estimate FLOP cost.

    This function sets the model to evaluation mode, computes the average 
    cross-entropy loss on the given validation dataset using teacher forcing, 
    and estimates the floating point operations (FLOPs) required for evaluation. 
    Optionally, evaluation can be limited to a subset of batches for faster runtime.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to evaluate. Should either be already prepared via `accelerator.prepare()`, 
        or will be prepared in this function if `accelerator` is None.

    val_dataset : torch.utils.data.Dataset
        The validation dataset containing tokenized input sequences.

    accelerator : accelerate.Accelerator or None
        Hugging Face `Accelerator` instance used to handle device placement and model wrapping.
        If None, a new `Accelerator` will be created and used.

    lora_ranks : int
        The rank of the LoRA decomposition, used for estimating LoRA-specific FLOPs.

    subset : int or None, optional
        If specified, limits evaluation to the first `subset` batches.

    print_summary : bool, default=False
        Whether to print a FLOPs summary after evaluation.

    Returns
    -------
    val_loss : float
        The average cross-entropy loss over the validation dataset (or subset).

    flop_cost : float
        The estimated number of FLOPs required to evaluate the model.
    """

    # Determine how many timesteps are being predicted (after inference split)
    prediction_point = round((1 - inference_split) * 100)
    print(f"Prediction horizon: {prediction_point} timesteps")

    # Lists to collect true and generated data for each sample
    prey_original = []
    predator_original = []
    prey_generated = []
    predator_generated = []
    total_flops = 0

    # Loop through a set number of random sequences from the dataset
    for i in range(no_random_samples):
        # Get model predictions and ground truth for this sequence
        prey_orig, predator_orig, prey_gen, predator_gen, flops = generate(
            model,
            tokeniser,
            dataset,
            sequence_no=i,
            inference_split=inference_split,
            plot=False,
            randomness=randomness, 
            print_output=False
        )
        total_flops += flops

        # Store only the predicted portion (last N timesteps)
        prey_original.append(prey_orig[-prediction_point:])
        predator_original.append(predator_orig[-prediction_point:])
        prey_generated.append(prey_gen[-prediction_point:])
        predator_generated.append(predator_gen[-prediction_point:])

    # Convert to NumPy arrays for element-wise operations
    error_prey = np.array(prey_original) - np.array(prey_generated)
    error_predator = np.array(predator_original) - np.array(predator_generated)

    # Compute Mean Absolute Error (MAE) per timestep, averaged over all sequences
    average_mean_abs_error_prey = np.mean(np.abs(error_prey), axis=0)
    average_mean_abs_error_predator = np.mean(np.abs(error_predator), axis=0)

    # Compute Mean Squared Error (MSE) per timestep, averaged over all sequences
    average_mean_squared_error_prey = np.mean(error_prey**2, axis=0)
    average_mean_squared_error_predator = np.mean(error_predator**2, axis=0)

    return (average_mean_abs_error_prey, average_mean_abs_error_predator,
        average_mean_squared_error_prey, average_mean_squared_error_predator), total_flops


def plot_model_errors(error_tuple,title_prefix="Model Prediction Errors"):
    """
    Plots MAE and MSE for prey and predator using a tuple of error arrays.

    Parameters
    ----------
    error_tuple : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
            - MAE for prey
            - MAE for predator
            - MSE for prey
            - MSE for predator
    title_prefix : str, optional
        Title prefix used for both plots.
    """
    mae_prey, mae_predator, mse_prey, mse_predator = error_tuple
    timesteps = np.arange(len(mae_prey))

    # ---- MAE Plot ----
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mae_prey, label="Prey MAE", marker="o", color="blue")
    plt.plot(timesteps, mae_predator, label="Predator MAE", marker="x", color="green")
    plt.title(f"{title_prefix} - MAE")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- MSE Plot ----
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mse_prey, label="Prey MSE", marker="o", color="red")
    plt.plot(timesteps, mse_predator, label="Predator MSE", marker="x", color="purple")
    plt.title(f"{title_prefix} - MSE")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()