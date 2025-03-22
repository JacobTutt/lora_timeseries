import torch
import torch.nn as nn
from .qwen import load_qwen_model

# ------------------------ LoRA Implementation ------------------------ #
class LoRALinear(nn.Module):
    """
    Implements Low-Rank Adaptation (LoRA) for a linear layer.

    This class wraps a standard nn.Linear layer with LoRA, which introduces two
    trainable low-rank matrices (A and B) while freezing the original layer's weights.
    The result is a parameter-efficient fine-tuning strategy that significantly reduces
    the number of trainable parameters — ideal for adapting large language models.

    Parameters
    ----------
    original_linear : nn.Linear
        The original linear layer to be adapted with LoRA.
    
    r : int
        Rank of the LoRA decomposition. Smaller values reduce trainable parameters.
    
    alpha : int, optional
        Scaling factor for LoRA output. Defaults to `r` if not provided, which
        ensures that the initial LoRA path has similar scale to the base layer.

    Attributes
    ----------
    original_linear : nn.Linear
        The frozen base linear layer.
    
    A : nn.Parameter
        Trainable low-rank matrix of shape `(r, in_features)`, used to project
        input down to a lower-dimensional space.
    
    B : nn.Parameter
        Trainable low-rank matrix of shape `(out_features, r)`, used to project
        back up to output space.

    alpha : int
        Scaling factor applied to the LoRA output. Balances its influence relative
        to the original (frozen) output.
    """

    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, nn.Linear), "LoRALinear expects an nn.Linear layer"
        
        # Store the frozen base layer, this is already done when importing the qwen model however is 
        # repeated here for robustness
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False 

        # Freeze bias if it exists, this does in the case of the Qwen model but if statement is here for robustness
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # Set the dimensions of these LoRA matrices based on the original layer and the rank
        # Inorder for the LoRA to work the dimensions once matrix multiplied must be the same as the original layer
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r  # Default scaling factor is r

        # Allocate LoRA matrices A and B on the same device as the base layer
        device = original_linear.weight.device

        # Initialise both A and B and use dimensions as defined above
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))     # Projects down
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))    # Projects up

        # Initialize A using He initialization (good for ReLU/linear activations)
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        """
        Define the forward pass through the LoRA-modified linear layer.

        This allows it to work with the rest of the model and be for backpropagation to be performed
        in training the rest of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features) after applying
            the base linear transformation and the scaled LoRA adaptation.
        """
        # Original frozen output
        base_out = self.original_linear(x)

        # LoRA adjustment: x → A^T → B^T
        lora_out = (x @ self.A.T) @ self.B.T

        # Combine base and LoRA output, scaled appropriately
        return base_out + lora_out * (self.alpha / self.r)


def full_model(lora_rank: int):
    """
    Load a pretrained Qwen model and inject Low-Rank Adaptation (LoRA)
    into its attention layers with the specified rank.

    This function wraps the `q_proj` and `v_proj` components of each self-attention
    block in the Qwen model with LoRALinear layers. If `lora_rank` is set to 0, 
    no modification is made and the base model is returned as-is.

    Parameters
    ----------
    lora_rank : int
        The rank of the LoRA decomposition. Must be a non-negative integer.
        - If 0, LoRA is not applied and the original model is returned.
        - If > 0, LoRALinear is injected into query and value projections.

    Returns
    -------
    model : transformers.PreTrainedModel
        The modified (or unmodified) Qwen model.

    tokeniser : transformers.PreTrainedTokenizer
        The tokenizer associated with the model.

    device : torch.device
        The device (CUDA, MPS, or CPU) the model is loaded onto.

    Raises
    ------
    ValueError
        If `lora_rank` is not an integer or is negative.
    """

    # Validate LoRA rank
    if not isinstance(lora_rank, int):
        raise ValueError(f"Expected lora_rank to be an int, got {type(lora_rank)}")
    if lora_rank < 0:
        raise ValueError(f"lora_rank must be non-negative, got {lora_rank}")

    # Load the frozen base Qwen model, tokeniser, and device
    model, tokeniser, device = load_qwen_model()

    # If LoRA rank is zero, skip modification
    if lora_rank == 0:
        print("Returned the base Qwen model without modification (rank = 0).")
        print(f"Model loaded on {device}")
        return model, tokeniser, device

    # Inject LoRA into Q and V projections of attention blocks
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank)

    print(f"Returning Qwen model injected with LoRA into Query and Value Projections with rank = {lora_rank}")
    print(f"Model loaded on {device}")
    
    return model, tokeniser, device