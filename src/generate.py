import torch
from .decoder import decoder
import matplotlib.pyplot as plt
from .flops_counter import model_generation_flops
import numpy as np


def plotting_trend(prey_original, prey_generated, predator_original, predator_generated, sequence_no, inference_split):
    """
    Plot original vs. generated population trends for prey and predators in a time series.

    Parameters
    ----------
    prey_original : list or np.ndarray
        Ground truth prey population over time.
    prey_generated : list or np.ndarray
        Model-generated prey population over time.
    predator_original : list or np.ndarray
        Ground truth predator population over time.
    predator_generated : list or np.ndarray
        Model-generated predator population over time.
    sequence_no : int
        Index of the sequence being visualized.
    inference_split : float
        Proportion (0 < x < 1) of the sequence used as input context.
    """
    timesteps = np.arange(len(prey_original))
    inference_timestep = int(inference_split * len(prey_original))

    plt.figure(figsize=(9, 6))

    # Plot prey populations
    plt.plot(timesteps, prey_original, label="True Prey", color="tab:blue", linewidth=2)
    plt.plot(timesteps, prey_generated, label="Predicted Prey", color="tab:red", linestyle="--", linewidth=2)

    # Plot predator populations
    plt.plot(timesteps, predator_original, label="True Predator", color="grey", linewidth=2)
    plt.plot(timesteps, predator_generated, label="Predicted Predator", color="green", linestyle="--", linewidth=2)

    # Mark inference start
    plt.axvline(inference_timestep, color="black", linestyle="--", linewidth=1.5, label="Start of Inference")
    plt.text(inference_timestep + 1, plt.ylim()[1]*0.95, 'Inference Start', color='black', fontsize=14, va='top')

    # Aesthetics
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Population (scaled)", fontsize=14)
    plt.title(f"Autoregressive Generation Preformance (System {sequence_no})", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc="upper left")
    plt.tight_layout()
    plt.show()

def generate(model, tokeniser, dataset, sequence_no = 0, inference_split = 0.8, lora_rank = 0, plot = False, randomness = False, print_output = False):
    """
    Generate future time series tokens from a given context using a trained transformer model.

    This function extracts one sequence from the dataset, tokenizes it, and uses the model to 
    generate the remainder of the sequence beyond the given inference split point. It optionally 
    visualizes the true and predicted prey/predator populations.

    Parameters
    ----------
    model : torch.nn.Module
        A LoRA-injected transformer model prepared for generation.

    tokeniser : transformers.PreTrainedTokenizer
        Tokenizer used for encoding/decoding the input and output sequences.

    dataset : list or Sequence[str]
        A list of time series strings (one per system), each representing prey and predator dynamics.

    sequence_no : int, optional
        Index of the time series sequence to generate from. Default is 0.

    inference_split : float, optional
        Fraction of the sequence to use as context before generation begins. Default is 0.8.

    lora_rank : int, optional
        LoRA rank used for FLOP cost estimation. Default is 0.

    plot : bool, optional
        If True, plots the original vs. generated sequences. Default is False.

    randomness : bool, optional
        If True, enables stochastic sampling (e.g., top-k, nucleus). If False, uses greedy decoding. Default is False.

    print_output : bool, optional
        If True, prints diagnostic messages including token counts and generation type. Default is False.

    Returns
    -------
    prey_original : list[float]
        Ground truth prey population values.

    predator_original : list[float]
        Ground truth predator population values.

    prey_generated : list[float]
        Model-generated prey population values.

    predator_generated : list[float]
        Model-generated predator population values.

    total_flops : float
        Estimated number of FLOPs used for generation.
    """

    model.eval()
    # Extract the desired sequence
    sequence = dataset[sequence_no]
    # Tokenise the sequence
    tokenised_input = tokeniser(sequence, return_tensors="pt")
    # Move the tokenised sequence to the device
    tokenised_sequence = tokenised_input.input_ids.to(model.device)
    attention_mask = tokenised_input.attention_mask.to(model.device)

    # split the index at the desired inference split - ie this is the point at which we will start generating
    split_indx = int(inference_split * tokenised_sequence.shape[1])
    input_tokens = tokenised_sequence[:, :split_indx]
    input_attention_mask = attention_mask[:, :split_indx]

    with torch.no_grad():
        # we need to account for the fact that in our scaling algorithm we scaled between 90% being between 0-10
        # this means some values above this have an extra figure and thus slightly more tokens are needed
        if print_output == True:
            print(f'Generating {1.2*(len(tokenised_input[0]) - split_indx)} tokens from a context of {len(input_tokens[0])} tokens')
        if print_output and randomness == True:
            print("Using a stochastic sampling technique, randomness is enabled")
        if print_output and randomness == False:
            print("Using a deterministic (greedy) sampling technique, randomness is disabled")
        output = model.generate(input_tokens, max_new_tokens = 1.2*(len(tokenised_input[0]) - split_indx),  attention_mask=input_attention_mask, do_sample=randomness)
        no_tokens_given = len(input_tokens[0])
        no_tokens_generated = len(output[0]) - len(input_tokens[0])
        total_flops, _ = model_generation_flops(no_tokens_given, no_tokens_generated, lora_ranks = lora_rank, randomness = randomness, print_summary = False)


    # Decode the generated text
    generated_qwen_text = tokeniser.decode(output[0], skip_special_tokens=True)
    prey_generated, predator_generated = decoder(generated_qwen_text)
    prey_original, predator_original = decoder(sequence)

    # Ensure that the generated sequence is the same length as the original
    prey_generated = prey_generated[:len(prey_original)]
    predator_generated = predator_generated[:len(predator_original)]

    if print_output == True:
        print(f'Flops associated with generation: {total_flops}')
 

    # Plot the results
    if plot == True:
        plotting_trend(prey_original, prey_generated, predator_original, predator_generated, sequence_no, inference_split)

    return  prey_original, predator_original, prey_generated, predator_generated, total_flops