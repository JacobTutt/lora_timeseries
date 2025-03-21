import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    """
    Loads the Qwen2.5 model and tokenizer with specified settings.

    Args:
        model_name (str): Name of the model on Hugging Face.

    Returns:
        tuple: (tokenizer, model, device)
    """
    print("Currently loading:", model_name)

    # Load the tokenizer with trust_remote_code=True
    tokeniser = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load the model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Dynamic selecting of device (GPU if available, else MPS for Mac, else CPU)
    # This should allow it to run on all devices - ie for assessment: Mac in my case
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Ensure LM head bias is trainable
    if model.lm_head.bias is None:
        model.lm_head.bias = torch.nn.Parameter(
            torch.zeros(model.config.vocab_size, device=device)
        )
    model.lm_head.bias.requires_grad = True

    print(f"Model loaded on {device}")

    return model, tokeniser, device


if __name__ == "__main__":  # Corrected line
    # Load model and check device
    model, tokeniser, device = load_qwen_model()