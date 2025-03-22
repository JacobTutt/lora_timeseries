import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Load the Qwen2.5 language model and tokenizer with appropriate device placement and training setup.

    This function fetches a pretrained Qwen2.5 model and tokeniser from Hugging Face, and places it on the
    optimal device (CUDA, Apple MPS, or CPU). 
    
    It also sets the model parameters up for efficient fine-tuning using LoRA by freezing the parameters 
    except the final language modeling (LM) head bias term.

    Parameters
    ----------
    model_name : str, optional
        The Hugging Face model name or path to load (default is "Qwen/Qwen2.5-0.5B-Instruct").

    Returns
    -------
    model : transformers.PreTrainedModel
        The loaded Qwen model configured for inference or LoRA fine-tuning.
    
    tokeniser : transformers.PreTrainedTokenizer
        The tokenizer corresponding to the Qwen model, with custom code trusted.

    device : torch.device
        The PyTorch device the model is placed on (CUDA, MPS, or CPU).
    """

    # Load the pretrained Qwen tokeniser, from hugging face
    tokeniser = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load the pretrained Qwen model, from hugging face 
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Dynamic selecting of device (GPU if available, else MPS for Mac, else CPU)
    # This should allow it to run on all devices and pick the best for each architecture - ie for assessment: Mac in my case
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Freeze all parameters so that no training will be preformed on the base model
    # We will only be training on the Low Rank Adaptation later applied and the bias of the logits
    for param in model.parameters():
        param.requires_grad = False

    # We ensure that a bias is applied to the logits term in vocab space to allow for training
    # These are the only part of the base model that will be trained along with the LoRA adaptation layers
    if model.lm_head.bias is None:
        model.lm_head.bias = torch.nn.Parameter(torch.zeros(model.config.vocab_size, device=device))
    model.lm_head.bias.requires_grad = True

    # Return device as well so eveything can be loaded onto this later
    return model, tokeniser, device


if __name__ == "__main__":
    # Load model and check device
    model, tokeniser, device = load_qwen_model()