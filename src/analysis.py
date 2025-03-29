import json
import numpy as np
import pandas as pd
from .generate import generate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython.display import display

def model_analysis(model, no_random_samples, tokeniser, dataset, inference_split=0.8, randomness=False, save_path=None):
    """
    Evaluate model predictive performance on a set of random time series samples and estimate FLOP cost.

    This function uses the `generate` method to simulate future trajectories of a predator-prey system
    using a trained model. It compares predicted values against ground truth over a specified prediction
    horizon and accumulates the associated floating point operations (FLOPs). The mean prediction error 
    and total FLOPs are returned, and optionally saved to disk.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be evaluated.

    no_random_samples : int
        Number of random time series sequences to sample from the dataset for evaluation.

    tokeniser : callable
        A tokenizer or sequence preparer compatible with the `generate` function for model input formatting.

    dataset : torch.utils.data.Dataset or similar
        The full dataset from which sequences are drawn for generation and evaluation.

    inference_split : float, optional, default=0.8
        Fraction of each sequence used as context for inference. The remainder is used for prediction evaluation.

    randomness : bool, optional, default=False
        Whether to include stochasticity in the prediction generation (e.g., sampling instead of greedy decoding).

    save_path : str or None, optional
        Path to save a JSON file containing the prediction errors and total FLOPs. If `None`, results are not saved.

    Returns
    -------
    error_prey : np.ndarray
        The error between predicted and true prey values over the prediction horizon 
        for all evaluated sequences (shape: [num_samples, prediction_length]).

    error_predator : np.ndarray
        The error between predicted and true predator values over the prediction horizon 
        for all evaluated sequences (shape: [num_samples, prediction_length]).

    total_generation_flops : float
        Estimated total number of floating point operations required to generate all predictions.
    """

    # Determine how many timesteps are being predicted (after inference split)
    prediction_point = round((1 - inference_split) * 100)
    print(f"Prediction horizon: {prediction_point} timesteps")

    # Lists to collect true and generated data for each sample
    prey_original = []
    predator_original = []
    prey_generated = []
    predator_generated = []
    total_generation_flops = 0

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
        total_generation_flops += flops

        # Store only the predicted portion (last N timesteps)
        prey_original.append(prey_orig[-prediction_point:])
        predator_original.append(predator_orig[-prediction_point:])
        prey_generated.append(prey_gen[-prediction_point:])
        predator_generated.append(predator_gen[-prediction_point:])

    # Save these arrays as lists in a json file with total_generation_flops
    if save_path is not None:
        save_dict = {
            "prey_original": np.array(prey_original).tolist(),
            "prey_generated": np.array(prey_generated).tolist(),
            "predator_original": np.array(predator_original).tolist(),
            "predator_generated": np.array(predator_generated).tolist(),
            "total_generation_flops": float(total_generation_flops)  # or int(), depending on type
        }
        with open(save_path, "w") as f:
            json.dump(save_dict, f)

    return prey_original, prey_generated, predator_original, predator_generated ,total_generation_flops





def plot_model_errors(prey_original, prey_generated, predator_original, predator_generated, draw_context = None, title_prefix="Model Prediction Errors"):
    """
    Visualize prediction errors of a time series model and display metric summary.

    This function computes and plots the Mean Absolute Error (MAE), Mean Squared Error (MSE),
    and Mean Absolute Percentage Error (MAPE) for both prey and predator time series predictions.
    Each metric is plotted over time with error bars representing the standard error of the mean.
    Zoomed-in insets show the first 10 timesteps. Additionally, a pandas DataFrame is printed that 
    summarizes the error metrics and their standard errors at each timestep.

    Parameters
    ----------
    prey_original : array-like
        Ground truth prey values over time for multiple sequences.

    prey_generated : array-like
        Model-generated prey predictions over time for the same sequences.

    predator_original : array-like
        Ground truth predator values over time for multiple sequences.

    predator_generated : array-like
        Model-generated predator predictions over time for the same sequences.

    title_prefix : str, optional
        Prefix to use for plot titles (default is "Model Prediction Errors").

    Returns
    -------
    None
        This function produces plots and prints a metric summary table to the console.
    """
    # Global plot settings
    AXIS_LABEL_SIZE = 14
    TICK_LABEL_SIZE = 12
    TITLE_SIZE = 16

    prey_original = np.array(prey_original)
    prey_generated = np.array(prey_generated)
    predator_original = np.array(predator_original)
    predator_generated = np.array(predator_generated)

    error_prey = prey_original - prey_generated
    error_predator = predator_original - predator_generated

    abs_error_prey = np.abs(error_prey)
    abs_error_predator = np.abs(error_predator)
    squared_error_prey = error_prey ** 2
    squared_error_predator = error_predator ** 2
    mape_prey_all = np.abs(error_prey / np.clip(prey_original, 1e-8, None)) * 100
    mape_predator_all = np.abs(error_predator / np.clip(predator_original, 1e-8, None)) * 100

    mae_prey = np.mean(abs_error_prey, axis=0)
    mae_predator = np.mean(abs_error_predator, axis=0)
    mse_prey = np.mean(squared_error_prey, axis=0)
    mse_predator = np.mean(squared_error_predator, axis=0)
    mape_prey = np.mean(mape_prey_all, axis=0)
    mape_predator = np.mean(mape_predator_all, axis=0)

    n = error_prey.shape[0]
    mae_prey_se = np.std(abs_error_prey, axis=0, ddof=1) / np.sqrt(n)
    mae_predator_se = np.std(abs_error_predator, axis=0, ddof=1) / np.sqrt(n)
    mse_prey_se = np.std(squared_error_prey, axis=0, ddof=1) / np.sqrt(n)
    mse_predator_se = np.std(squared_error_predator, axis=0, ddof=1) / np.sqrt(n)
    mape_prey_se = np.std(mape_prey_all, axis=0, ddof=1) / np.sqrt(n)
    mape_predator_se = np.std(mape_predator_all, axis=0, ddof=1) / np.sqrt(n)

    timesteps = np.arange(len(mae_prey))

    def add_inset(ax, x, y1, yerr1, y2, yerr2, color1, color2):
        axins = inset_axes(ax, width="40%", height="40%", loc='upper left')
        axins.errorbar(x[:10], y1[:10], yerr=yerr1[:10], fmt='-o', color=color1, capsize=3, label="Prey")
        axins.errorbar(x[:10], y2[:10], yerr=yerr2[:10], fmt='-x', color=color2, capsize=3, label="Predator")
        axins.set_xlim(0, 9)
        axins.grid(True, linestyle="--", alpha=0.4)
        axins.tick_params(labelsize=10, direction='in', length=4)
        axins.yaxis.tick_right()
        axins.yaxis.set_label_position("right")

    # --- MAE ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(timesteps, mae_prey, yerr=mae_prey_se, label="Prey MAE", fmt="-o", color="blue", capsize=3)
    ax.errorbar(timesteps, mae_predator, yerr=mae_predator_se, label="Predator MAE", fmt="-x", color="green", capsize=3)
    ax.set_title(f"{title_prefix} - MAE", fontsize=TITLE_SIZE)
    ax.set_xlabel("Timestep", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean Absolute Error", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.6)
    if draw_context is not None:
        ax.axvline(draw_context, color='red', linestyle='--', linewidth=2, label = 'Trained Context Length')
    ax.legend(loc='upper right', fontsize=12)
    add_inset(ax, timesteps, mae_prey, mae_prey_se, mae_predator, mae_predator_se, "blue", "green")
    plt.tight_layout()
    plt.show()

    # --- MSE ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(timesteps, mse_prey, yerr=mse_prey_se, label="Prey MSE", fmt="-o", color="red", capsize=3)
    ax.errorbar(timesteps, mse_predator, yerr=mse_predator_se, label="Predator MSE", fmt="-x", color="purple", capsize=3)
    ax.set_title(f"{title_prefix} - MSE", fontsize=TITLE_SIZE)
    ax.set_xlabel("Timestep", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean Squared Error", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.6)
    if draw_context is not None:
        ax.axvline(draw_context, color='red', linestyle='--', linewidth=2, label = 'Trained Context Length')
    ax.legend(loc='upper right', fontsize=12)
    add_inset(ax, timesteps, mse_prey, mse_prey_se, mse_predator, mse_predator_se, "red", "purple")
    plt.tight_layout()
    plt.show()

    # --- MAPE ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(timesteps, mape_prey, yerr=mape_prey_se, label="Prey MAPE", fmt="-o", color="darkorange", capsize=3)
    ax.errorbar(timesteps, mape_predator, yerr=mape_predator_se, label="Predator MAPE", fmt="-x", color="teal", capsize=3)
    ax.set_title(f"{title_prefix} - MAPE", fontsize=TITLE_SIZE)
    ax.set_xlabel("Timestep", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean Absolute Percentage Error (%)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.6)
    if draw_context is not None:
        ax.axvline(draw_context, color='red', linestyle='--', linewidth=2, label = 'Trained Context Length')
    ax.legend(loc='upper right', fontsize=12)
    add_inset(ax, timesteps, mape_prey, mape_prey_se, mape_predator, mape_predator_se, "darkorange", "teal")
    plt.tight_layout()
    plt.show()

    # --- Metric Table ---
    df = pd.DataFrame({
        "Timestep": timesteps,
        "MAE Prey ± SE": [f"{m:.3f} ± {e:.3f}" for m, e in zip(mae_prey, mae_prey_se)],
        "MAE Predator ± SE": [f"{m:.3f} ± {e:.3f}" for m, e in zip(mae_predator, mae_predator_se)],
        "MSE Prey ± SE": [f"{m:.3f} ± {e:.3f}" for m, e in zip(mse_prey, mse_prey_se)],
        "MSE Predator ± SE": [f"{m:.3f} ± {e:.3f}" for m, e in zip(mse_predator, mse_predator_se)],
        "MAPE Prey ± SE (%)": [f"{m:.2f} ± {e:.2f}" for m, e in zip(mape_prey, mape_prey_se)],
        "MAPE Predator ± SE (%)": [f"{m:.2f} ± {e:.2f}" for m, e in zip(mape_predator, mape_predator_se)],
    })

    display(df)

def exponential_moving_average(values, beta):
    """Compute time-weighted exponential moving average."""
    smoothed = []
    avg = 0
    for i, v in enumerate(values):
        avg = beta * avg + (1 - beta) * v
        corrected_avg = avg / (1 - beta**(i + 1))  # Bias correction
        smoothed.append(corrected_avg)
    return np.array(smoothed)

def hyperparam_analysis(file_paths, labels, ema_beta=0.9, bar_chart = False, log_scale = False):
    """
    Compares hyperparameter runs by plotting training/validation losses and showing performance summary.

    Parameters
    ----------
    file_paths : list of str
        Paths to JSON log files.
    labels : list of str
        Experiment labels.
    ema_beta : float
        EMA smoothing coefficient for training loss (0 < beta < 1).
    """
    assert len(file_paths) == len(labels), "Each file must have a corresponding label."

    # ---- Global Plot Style Parameters ----
    labelsize = 15
    ticksize = 14
    titlesize = 15
    linewidth = 2
    alpha_raw = 0.5
    color_palette = plt.get_cmap("Set1")

    final_val_losses = []
    table_rows = []

    # Side-by-side training and validation loss plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_train, ax_val = axes

    for idx, (path, label) in enumerate(zip(file_paths, labels)):
        with open(path, "r") as f:
            data = json.load(f)

        val_steps = data["val_step_tracker"]
        train_steps = data["train_step_tracker"]
        val_loss = data["val_loss_tracker"]
        train_loss = data["train_loss_tracker"]
        final_val_loss = data["val_loss_final"]
        train_flops = data["training_flops"]
        eval_flops = data["total_eval_cost"]
        total_flops = train_flops + eval_flops

        color = color_palette(idx % 10)

        # ---- Training Plot (Raw + Smoothed) ----
        ax_train.plot(train_steps, train_loss, label=f"{label}", color=color, alpha=alpha_raw)
        smoothed_train = exponential_moving_average(train_loss, ema_beta)
        ax_train.plot(train_steps, smoothed_train, color=color, linewidth=linewidth)

        # ---- Validation Plot ----
        ax_val.plot(val_steps, val_loss, label=f"{label} (Final Val Loss: {final_val_loss:.4f})", color=color, linestyle="--", linewidth=linewidth)

        # Collect for final summary table
        final_val_losses.append(final_val_loss)
        table_rows.append({
            "Label": label,
            "Train FLOPs": train_flops,
            "Eval FLOPs": eval_flops,
            "Total FLOPs": total_flops,
            "Final Val Loss": final_val_loss
        })

    # ---- Formatting Training Plot ----
    ax_train.set_title("Training Loss", fontsize=titlesize)
    ax_train.set_xlabel("Step", fontsize=labelsize)
    if log_scale:
        ax_train.set_ylabel("Log Cross Entropy Loss", fontsize=labelsize)
    else:
        ax_train.set_ylabel("Cross Entropy Loss", fontsize=labelsize)
    ax_train.tick_params(labelsize=ticksize)
    ax_train.grid(True, linestyle="--", alpha=0.6)
    ax_train.legend(fontsize=10, loc = "upper right")
    if log_scale:
        ax_train.set_yscale('log')

    # ---- Formatting Validation Plot ----

    ax_val.set_title("Validation Loss", fontsize=titlesize)

    ax_val.set_xlabel("Step", fontsize=labelsize)

    ax_val.tick_params(labelsize=ticksize)
    ax_val.grid(True, linestyle="--", alpha=0.6)
    ax_val.legend(fontsize=10, loc = "upper right")
    if log_scale:
        ax_val.set_yscale('log')

    fig.tight_layout()
    plt.show()
    if bar_chart:
        # ---- Final Validation Loss Bar Chart ----
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        bar_colors = [color_palette(i % 10) for i in range(len(labels))]
        ax_bar.bar(labels, final_val_losses, color=bar_colors)
        ax_bar.set_title("Final Validation Loss", fontsize=titlesize)
        ax_bar.set_ylabel("Loss", fontsize=labelsize)
        ax_bar.set_xticks(range(len(labels)))
        ax_bar.set_xticklabels(labels, rotation=15, fontsize=ticksize)
        ax_bar.tick_params(labelsize=ticksize)
        fig_bar.tight_layout()
        plt.show()

    # ---- Summary Table ----
    df = pd.DataFrame(table_rows)
    display(df)



def model_train_plot(train_step_tracker, train_loss_tracker, val_step_tracker, val_loss_tracker, val_loss_final, early_stopping_step=None, ema_beta=0.9, log_scale = False):
    """
    Plot the training and validation loss curves over training steps with optional exponential moving average smoothing.

    This function visualizes the training and validation loss progression during model training.
    The raw training loss is shown as a faded curve, while a smoothed version using an exponential
    moving average is overlaid. The final validation loss is annotated, and an early stopping step
    is indicated if provided.

    Parameters
    ----------
    train_step_tracker : list or np.ndarray
        Steps at which training loss was recorded.

    train_loss_tracker : list or np.ndarray
        Raw training loss values at each recorded step.

    val_step_tracker : list or np.ndarray
        Steps at which validation loss was evaluated.

    val_loss_tracker : list or np.ndarray
        Validation loss values at each evaluation step.

    val_loss_final : float
        Final validation loss to be displayed in the legend.

    early_stopping_step : int, optional
        Step number where early stopping was triggered, if applicable.

    ema_beta : float, default=0.9
        Smoothing factor used for exponential moving average of training loss.

    Returns
    -------
    None
        Displays the training/validation loss plot.
    """
    # Compute smoothed training loss
    ema_train_loss = exponential_moving_average(train_loss_tracker, beta=ema_beta)

    # Plot
    plt.figure(figsize=(10, 6))

    # Training loss — faded blue
    plt.plot(train_step_tracker, train_loss_tracker, color="#1f77b4", linewidth=2, alpha=0.4)

    # Smoothed training loss — solid blue
    plt.plot(train_step_tracker, ema_train_loss, color="#1f77b4", linewidth=2.5, label="Training Loss")

    # Validation loss — solid red
    plt.plot(val_step_tracker, val_loss_tracker, color="#d62728", linewidth=2.5, label=f"Validation Loss (final = {val_loss_final:.4f})")

    # Early stopping
    if early_stopping_step:
        plt.axvline(x=early_stopping_step, color='blue', linestyle='--', linewidth=3, label=f"Early Stopping: Step {early_stopping_step}")

    # Titles and labels
    plt.title("Training and Validation Loss", fontsize=18, weight='bold')
    plt.xlabel("Training Step", fontsize=15)
    plt.ylabel("Cross-Entropy Loss", fontsize=15)

    # Axis ticks and limits
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(left=0)
    if log_scale:
        plt.yscale('log')
    # Grid
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

    # Legend
    plt.legend(fontsize=13, loc='upper left')

    # Layout
    plt.tight_layout()
    plt.show()


def plot_model_errors(prey_original, prey_generated, predator_original, predator_generated, draw_context = None, title_prefix="Model Prediction Errors"):
    """
    Visualize prediction errors of a time series model and display metric summary.

    This function computes and plots the Mean Absolute Error (MAE), Mean Squared Error (MSE),
    and Mean Absolute Percentage Error (MAPE) for both prey and predator time series predictions.
    Each metric is plotted over time with error bars representing the standard error of the mean.
    Zoomed-in insets show the first 10 timesteps. Additionally, a pandas DataFrame is printed that 
    summarizes the error metrics and their standard errors at each timestep.

    Parameters
    ----------
    prey_original : array-like
        Ground truth prey values over time for multiple sequences.

    prey_generated : array-like
        Model-generated prey predictions over time for the same sequences.

    predator_original : array-like
        Ground truth predator values over time for multiple sequences.

    predator_generated : array-like
        Model-generated predator predictions over time for the same sequences.

    title_prefix : str, optional
        Prefix to use for plot titles (default is "Model Prediction Errors").

    Returns
    -------
    None
        This function produces plots and prints a metric summary table to the console.
    """
    # Global plot settings
    AXIS_LABEL_SIZE = 14
    TICK_LABEL_SIZE = 12
    TITLE_SIZE = 16

    prey_original = np.array(prey_original)
    prey_generated = np.array(prey_generated)
    predator_original = np.array(predator_original)
    predator_generated = np.array(predator_generated)

    error_prey = prey_original - prey_generated
    error_predator = predator_original - predator_generated

    abs_error_prey = np.abs(error_prey)
    abs_error_predator = np.abs(error_predator)
    squared_error_prey = error_prey ** 2
    squared_error_predator = error_predator ** 2
    mape_prey_all = np.abs(error_prey / np.clip(prey_original, 1e-8, None)) * 100
    mape_predator_all = np.abs(error_predator / np.clip(predator_original, 1e-8, None)) * 100

    mae_prey = np.mean(abs_error_prey, axis=0)
    mae_predator = np.mean(abs_error_predator, axis=0)
    mse_prey = np.mean(squared_error_prey, axis=0)
    mse_predator = np.mean(squared_error_predator, axis=0)
    mape_prey = np.mean(mape_prey_all, axis=0)
    mape_predator = np.mean(mape_predator_all, axis=0)

    n = error_prey.shape[0]
    mae_prey_se = np.std(abs_error_prey, axis=0, ddof=1) / np.sqrt(n)
    mae_predator_se = np.std(abs_error_predator, axis=0, ddof=1) / np.sqrt(n)
    mse_prey_se = np.std(squared_error_prey, axis=0, ddof=1) / np.sqrt(n)
    mse_predator_se = np.std(squared_error_predator, axis=0, ddof=1) / np.sqrt(n)
    mape_prey_se = np.std(mape_prey_all, axis=0, ddof=1) / np.sqrt(n)
    mape_predator_se = np.std(mape_predator_all, axis=0, ddof=1) / np.sqrt(n)

    timesteps = np.arange(len(mae_prey))

    def add_inset(ax, x, y1, yerr1, y2, yerr2, color1, color2):
        axins = inset_axes(ax, width="40%", height="40%", loc='upper left')
        axins.errorbar(x[:10], y1[:10], yerr=yerr1[:10], fmt='-o', color=color1, capsize=3, label="Prey")
        axins.errorbar(x[:10], y2[:10], yerr=yerr2[:10], fmt='-x', color=color2, capsize=3, label="Predator")
        axins.set_xlim(0, 9)
        axins.grid(True, linestyle="--", alpha=0.4)
        axins.tick_params(labelsize=10, direction='in', length=4)
        axins.yaxis.tick_right()
        axins.yaxis.set_label_position("right")

    # --- MAE ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(timesteps, mae_prey, yerr=mae_prey_se, label="Prey MAE", fmt="-o", color="blue", capsize=3)
    ax.errorbar(timesteps, mae_predator, yerr=mae_predator_se, label="Predator MAE", fmt="-x", color="green", capsize=3)
    ax.set_title(f"{title_prefix} - MAE", fontsize=TITLE_SIZE)
    ax.set_xlabel("Timestep", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean Absolute Error", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.6)
    if draw_context is not None:
        ax.axvline(draw_context, color='red', linestyle='--', linewidth=2, label = 'Trained Context Length')
    ax.legend(loc='upper right', fontsize=12)
    add_inset(ax, timesteps, mae_prey, mae_prey_se, mae_predator, mae_predator_se, "blue", "green")
    plt.tight_layout()
    plt.show()

    # --- MSE ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(timesteps, mse_prey, yerr=mse_prey_se, label="Prey MSE", fmt="-o", color="red", capsize=3)
    ax.errorbar(timesteps, mse_predator, yerr=mse_predator_se, label="Predator MSE", fmt="-x", color="purple", capsize=3)
    ax.set_title(f"{title_prefix} - MSE", fontsize=TITLE_SIZE)
    ax.set_xlabel("Timestep", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean Squared Error", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.6)
    if draw_context is not None:
        ax.axvline(draw_context, color='red', linestyle='--', linewidth=2, label = 'Trained Context Length')
    ax.legend(loc='upper right', fontsize=12)
    add_inset(ax, timesteps, mse_prey, mse_prey_se, mse_predator, mse_predator_se, "red", "purple")
    plt.tight_layout()
    plt.show()

    # --- MAPE ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(timesteps, mape_prey, yerr=mape_prey_se, label="Prey MAPE", fmt="-o", color="darkorange", capsize=3)
    ax.errorbar(timesteps, mape_predator, yerr=mape_predator_se, label="Predator MAPE", fmt="-x", color="teal", capsize=3)
    ax.set_title(f"{title_prefix} - MAPE", fontsize=TITLE_SIZE)
    ax.set_xlabel("Timestep", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean Absolute Percentage Error (%)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
    ax.grid(True, linestyle="--", alpha=0.6)
    if draw_context is not None:
        ax.axvline(draw_context, color='red', linestyle='--', linewidth=2, label = 'Trained Context Length')
    ax.legend(loc='upper right', fontsize=12)
    add_inset(ax, timesteps, mape_prey, mape_prey_se, mape_predator, mape_predator_se, "darkorange", "teal")
    plt.tight_layout()
    plt.show()

    # --- Metric Table ---
    df = pd.DataFrame({
        "Timestep": timesteps,
        "MAE Prey ± SE": [f"{m:.3f} ± {e:.3f}" for m, e in zip(mae_prey, mae_prey_se)],
        "MAE Predator ± SE": [f"{m:.3f} ± {e:.3f}" for m, e in zip(mae_predator, mae_predator_se)],
        "MSE Prey ± SE": [f"{m:.3f} ± {e:.3f}" for m, e in zip(mse_prey, mse_prey_se)],
        "MSE Predator ± SE": [f"{m:.3f} ± {e:.3f}" for m, e in zip(mse_predator, mse_predator_se)],
        "MAPE Prey ± SE (%)": [f"{m:.2f} ± {e:.2f}" for m, e in zip(mape_prey, mape_prey_se)],
        "MAPE Predator ± SE (%)": [f"{m:.2f} ± {e:.2f}" for m, e in zip(mape_predator, mape_predator_se)],
    })

    display(df)


def plot_model_errors_compare(prey_original_trained, predator_original_trained, prey_generated_trained,
    predator_generated_trained, prey_original_untrained, predator_original_untrained,
    prey_generated_untrained, predator_generated_untrained, draw_context=None,
    title_prefix="Model Comparison Errors", scale=True):
    """
    Compare and visualize prediction errors for trained and untrained time series models.

    This function evaluates and compares model performance for two sets of predictions:
    one from a trained model and one from an untrained model. It computes the following
    metrics over time for both prey and predator time series:
    
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Mean Absolute Percentage Error (MAPE)

    The function plots the trained model errors with standard error bars and overlays
    the untrained model's performance as faint background curves. Additionally, it
    includes zoomed-in insets for the first 10 time steps for each metric and 
    outputs a table comparing the trained and untrained metrics at each time step,
    including standard error and percentage improvement over the untrained model.

    Parameters
    ----------
    prey_original_trained : array-like of shape (n_samples, n_timesteps)
        Ground truth prey values used to evaluate the trained model.

    predator_original_trained : array-like of shape (n_samples, n_timesteps)
        Ground truth predator values used to evaluate the trained model.

    prey_generated_trained : array-like of shape (n_samples, n_timesteps)
        Predictions of prey made by the trained model.

    predator_generated_trained : array-like of shape (n_samples, n_timesteps)
        Predictions of predator made by the trained model.

    prey_original_untrained : array-like of shape (n_samples, n_timesteps)
        Ground truth prey values used to evaluate the untrained model.

    predator_original_untrained : array-like of shape (n_samples, n_timesteps)
        Ground truth predator values used to evaluate the untrained model.

    prey_generated_untrained : array-like of shape (n_samples, n_timesteps)
        Predictions of prey made by the untrained model.

    predator_generated_untrained : array-like of shape (n_samples, n_timesteps)
        Predictions of predator made by the untrained model.

    draw_context : int, optional
        If provided, a vertical red dashed line will mark the training context length
        on each plot.

    title_prefix : str, default="Model Comparison Errors"
        Prefix used in the titles of all plots.

    Returns
    -------
    None
        The function displays plots of error metrics and prints a pandas DataFrame
        summarizing each metric per timestep with standard error and percentage
        improvement from the untrained to the trained model.
    """


    def compute_metrics(true, pred):
        err = true - pred
        abs_err = np.abs(err)
        sq_err = err**2
        mape = np.abs(err / np.clip(true, 1e-8, None)) * 100
        return {
            "mae": np.mean(abs_err, axis=0),
            "mae_se": np.std(abs_err, axis=0, ddof=1) / np.sqrt(len(err)),
            "mse": np.mean(sq_err, axis=0),
            "mse_se": np.std(sq_err, axis=0, ddof=1) / np.sqrt(len(err)),
            "mape": np.mean(mape, axis=0),
            "mape_se": np.std(mape, axis=0, ddof=1) / np.sqrt(len(err)),
        }

    def add_inset(ax, x,
                y1, yerr1, y2, yerr2,           # trained
                y1_bg, yerr1_bg, y2_bg, yerr2_bg, # untrained
                c1, c2,
                scale):
        axins = inset_axes(ax, width="40%", height="40%", loc='upper left')

        # Untrained (faded)
        axins.errorbar(x[:10], y1_bg[:10], yerr=yerr1_bg[:10], fmt='--o', color=c1, capsize=3, alpha=0.3)
        axins.errorbar(x[:10], y2_bg[:10], yerr=yerr2_bg[:10], fmt='--x', color=c2, capsize=3, alpha=0.3)

        # Trained
        axins.errorbar(x[:10], y1[:10], yerr=yerr1[:10], fmt='-o', color=c1, capsize=3)
        axins.errorbar(x[:10], y2[:10], yerr=yerr2[:10], fmt='-x', color=c2, capsize=3)

        axins.set_xlim(0, 9)

        if scale:
            all_trained_inset = np.concatenate([
                y1[:10] + yerr1[:10],
                y1[:10] - yerr1[:10],
                y2[:10] + yerr2[:10],
                y2[:10] - yerr2[:10]
            ])
            axins.set_ylim(bottom=np.min(all_trained_inset) * 0.95,
                        top=np.max(all_trained_inset) * 1.05)

        axins.grid(True, linestyle="--", alpha=0.4)
        axins.tick_params(labelsize=10, direction='in', length=4)
        axins.yaxis.tick_right()
        axins.yaxis.set_label_position("right")

    # Convert to arrays
    prey_original_trained = np.array(prey_original_trained)
    prey_generated_trained = np.array(prey_generated_trained)
    predator_original_trained = np.array(predator_original_trained)
    predator_generated_trained = np.array(predator_generated_trained)
    prey_original_untrained = np.array(prey_original_untrained)
    prey_generated_untrained = np.array(prey_generated_untrained)
    predator_original_untrained = np.array(predator_original_untrained)
    predator_generated_untrained = np.array(predator_generated_untrained)

    metrics_trained_prey = compute_metrics(prey_original_trained, prey_generated_trained)
    metrics_trained_pred = compute_metrics(predator_original_trained, predator_generated_trained)
    metrics_untrained_prey = compute_metrics(prey_original_untrained, prey_generated_untrained)
    metrics_untrained_pred = compute_metrics(predator_original_untrained, predator_generated_untrained)

    timesteps = np.arange(len(metrics_trained_prey["mae"]))

    def plot_metric(metric_name, ylabel, color1, color2, scale=True):
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot untrained (faint)
        ax.errorbar(timesteps, metrics_untrained_prey[metric_name],
                    yerr=metrics_untrained_prey[f"{metric_name}_se"],
                    fmt='-o', color=color1, capsize=3, alpha=0.3, label="Prey (untrained)")
        ax.errorbar(timesteps, metrics_untrained_pred[metric_name],
                    yerr=metrics_untrained_pred[f"{metric_name}_se"],
                    fmt='-x', color=color2, capsize=3, alpha=0.3, label="Predator (untrained)")

        # Plot trained
        ax.errorbar(timesteps, metrics_trained_prey[metric_name],
                    yerr=metrics_trained_prey[f"{metric_name}_se"],
                    fmt='-o', color=color1, capsize=3, label="Prey (trained)")
        ax.errorbar(timesteps, metrics_trained_pred[metric_name],
                    yerr=metrics_trained_pred[f"{metric_name}_se"],
                    fmt='-x', color=color2, capsize=3, label="Predator (trained)")

        # Apply axis scaling based only on trained data if requested
        if scale:
            all_trained = np.concatenate([
                metrics_trained_prey[metric_name],
                metrics_trained_pred[metric_name] +
                metrics_trained_prey[f"{metric_name}_se"],
                metrics_trained_pred[f"{metric_name}_se"]
            ])
            ax.set_ylim(bottom=0, top=np.max(all_trained) * 1.3)

        ax.set_title(f"{title_prefix} - {metric_name.upper()}", fontsize=16)
        ax.set_xlabel("Timestep", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        if draw_context is not None:
            ax.axvline(draw_context, color='red', linestyle='--', linewidth=2, label="Trained Context Length")
        ax.legend(loc='upper right', fontsize=12)

        add_inset(
            ax, timesteps,
            metrics_trained_prey[metric_name], metrics_trained_prey[f"{metric_name}_se"],
            metrics_trained_pred[metric_name], metrics_trained_pred[f"{metric_name}_se"],
            metrics_untrained_prey[metric_name], metrics_untrained_prey[f"{metric_name}_se"],
            metrics_untrained_pred[metric_name], metrics_untrained_pred[f"{metric_name}_se"],
            color1, color2, scale
        )
        plt.tight_layout()
        plt.show()

    plot_metric("mae", "Mean Absolute Error", "blue", "green", scale=scale)
    plot_metric("mse", "Mean Squared Error", "red", "purple", scale=scale)
    plot_metric("mape", "Mean Absolute Percentage Error (%)", "darkorange", "teal", scale=scale)
    df = pd.DataFrame({
        "Timestep": timesteps,
        "MAE Prey": [f"{m:.3f} ± {se:.3f} ({100*(u-m)/u:.1f}%)" for m, se, u in zip(metrics_trained_prey["mae"], metrics_trained_prey["mae_se"], metrics_untrained_prey["mae"])],
        "MAE Predator": [f"{m:.3f} ± {se:.3f} ({100*(u-m)/u:.1f}%)" for m, se, u in zip(metrics_trained_pred["mae"], metrics_trained_pred["mae_se"], metrics_untrained_pred["mae"])],
        "MSE Prey": [f"{m:.3f} ± {se:.3f} ({100*(u-m)/u:.1f}%)" for m, se, u in zip(metrics_trained_prey["mse"], metrics_trained_prey["mse_se"], metrics_untrained_prey["mse"])],
        "MSE Predator": [f"{m:.3f} ± {se:.3f} ({100*(u-m)/u:.1f}%)" for m, se, u in zip(metrics_trained_pred["mse"], metrics_trained_pred["mse_se"], metrics_untrained_pred["mse"])],
        "MAPE Prey (%)": [f"{m:.2f} ± {se:.2f} ({100*(u-m)/u:.1f}%)" for m, se, u in zip(metrics_trained_prey["mape"], metrics_trained_prey["mape_se"], metrics_untrained_prey["mape"])],
        "MAPE Predator (%)": [f"{m:.2f} ± {se:.2f} ({100*(u-m)/u:.1f}%)" for m, se, u in zip(metrics_trained_pred["mape"], metrics_trained_pred["mape_se"], metrics_untrained_pred["mape"])],
    })

    display(df)