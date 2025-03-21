import math
import numpy as np

# GLOBAL VARIABLES
# Cost parameters for each type of mathematical operation
flops_cost_addition = 1
flops_cost_multiplication = 1 
flops_cost_division = 1
flops_cost_exponentiation = 10
flops_cost_sqrt = 10

# Rank of the Lora model
lora_rank = 4         


def convert_tokens_to_embeddings():
    """
    Determines the number of flops required to convert tokens to embeddings

    This is simply a memory look up operation and does not require any flops in the forwards pass
    """
    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [0, 0, 0, 0, 0]
    return np.array(results_array)

def flops_positional_embedding(no_tokens, embedding_dim):
    """
    Compute the flops for a single data element in a positional embedding layer

    This is a simple addition operation at the beginning of the embegging space which adds a positional embedding 
    to every value in th embedding input space (no. tokens x dimension fo embedding space)
    """
    additions = no_tokens * embedding_dim

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, 0*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]
    return np.array(results_array)


def query_head_flops(no_tokens, embedding_dim, query_heads):
    """
    In this we calculate the number of flops in generating the query matrix for the attention mechanism

    In the Qwen model we have 14 of these heads so the dimemnsionality of the embedding space is split across
    these heads. Ie dim_query_head. However in the forward pass we simply are then applying this to 14 heads and thus the 
    flops is no different across all heads as it would be for a single head. 

    This calculation is broken down excplicitly for each head for clarity.
    """

    # At first we will takle each heads calculations seperately
    # The number of dimensions per head - 
    dim_query_head = embedding_dim // query_heads  # Dimension per query head - will be 64 but laid out explicitly
    # Multiplications per head
    multiplications_head = no_tokens * embedding_dim * dim_query_head
    # Additions per head
    additions_head = no_tokens * (embedding_dim -1) * dim_query_head 
    # There is also an additional bias term that is added to the attention heads - described by the qwen architecture
    additions_head += no_tokens * dim_query_head
    # Sum over all these heads
    additions = additions_head * query_heads
    multiplications = multiplications_head * query_heads

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]

    return np.array(results_array)

def key_head_flops(no_tokens, embedding_dim, key_heads):
    """
    In this we calculate the number of flops in generating the key matrix for the attention mechanism

    In the Qwen model we have 2 of these heads so the dimemnsionality of the embedding space is split across
    these heads. Ie dim_query_head. However in the forward pass we simply are then repeating the caluclation for
    the 2 heads and thus the flops is no different across all heads as it would be for a single head. 

    This calculation is broken down excplicitly for each head for clarity.
    """

    # At first we will takle each heads calculations seperately
    # The number of dimensions per head - 
    dim_key_head = embedding_dim // key_heads  # Dimension per query head - will be 64 but laid out explicitly
    # Multiplications per head
    multiplications_head = no_tokens * embedding_dim * dim_key_head
    # Additions per head
    additions_head = no_tokens * (embedding_dim -1 ) * dim_key_head 
    # There is also an additional bias term that is added to the attention heads - described by the qwen architecture
    additions_head += no_tokens * dim_key_head
    # Sum over all these heads
    additions = additions_head * key_heads
    multiplications = multiplications_head * key_heads

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]

    return np.array(results_array)

def value_head_flops(no_tokens, embedding_dim, value_heads):
    """
    In this we calculate the number of flops in generating the value matrix for the attention mechanism

    In the Qwen model we have 2 of these heads so the dimemnsionality of the embedding space is split across
    these heads. Ie dim_query_head. However in the forward pass we simply are then repeating the caluclation for
    the 2 heads and thus the flops is no different across all heads as it would be for a single head.  

    This calculation is broken down excplicitly for each head for clarity.
    """

    # At first we will takle each heads calculations seperately
    # The number of dimensions per head - 
    dim_value_head = embedding_dim // value_heads  # Dimension per query head - will be 64 but laid out explicitly
    # Multiplications per head
    multiplications_head = no_tokens * embedding_dim * dim_value_head
    # Additions per head
    additions_head = no_tokens * (embedding_dim -1 ) * dim_value_head 
    # There is also an additional bias term that is added to the attention heads - described by the qwen architecture
    additions_head += no_tokens * dim_value_head
    # Sum over all these heads
    additions = additions_head * value_heads
    multiplications = multiplications_head * value_heads

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]

    return np.array(results_array)


def attention_mechanism_flops(no_tokens, embedding_dim, query_heads):
    """
    In this we calculate the number of flops in the attention mechanism to work out relations in queries and 
    keys. 

    This is preformed for each head - there is some ambiguity in the qwen model as this it uses Grouped Query Attention, 
    the follow calculations are based of the mathematics of Multi-Head Attention with 14 query heads, and thus 14 seperate 
    attentions matrices are calculated. 
    
    This is broken down for each head for clarity.
    """
    
    # The number of dimensions per head - 
    dim_query_head = embedding_dim // query_heads  # Dimension per query head - will be 64 but laid out explicitly
    # Multiplications per head
    multiplications_head = no_tokens * no_tokens * dim_query_head
    # Additions per head
    additions_head = no_tokens * no_tokens * (dim_query_head - 1) 
    # The whole matrix is also normalised by dividing by the dimensionality of the query and keys
    square_root = 1 # Can be reused for all heads and stored i memeory - as
    divisions_head = no_tokens * no_tokens
    # Sum over all these heads
    additions = additions_head * query_heads
    multiplications = multiplications_head * query_heads
    divisions = divisions_head * query_heads

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, divisions*flops_cost_division, 0*flops_cost_exponentiation, square_root*flops_cost_sqrt]
 
    return np.array(results_array)


def softmax_flops(no_tokens, query_heads):
    """
    In this we calculate the number of flops in the softmax operation in the attention mechanism for each head 
    and combine this across all heads

    Each attention head is of the dimenions (n_head * n_head) and for each we calculate the exponential and sum across
    the rows for normalisation. We then divide all terms by its respective row sum.
    """
    # Calculate each - sum and then normalise through division
    exponentiations_head = no_tokens * no_tokens 
    additions_head = no_tokens * (no_tokens - 1)
    divisions_head = no_tokens * no_tokens

    exponentiations = exponentiations_head * query_heads
    additions = additions_head * query_heads
    divisions = divisions_head * query_heads

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, 0*flops_cost_multiplication, divisions*flops_cost_division, exponentiations*flops_cost_exponentiation, 0*flops_cost_sqrt]
    return np.array(results_array)

def softmax_values_flops(no_tokens, embedding_dim, value_heads, query_heads):
    """
    In this we calculate combining the weighted sum of the values with the attention, this is were the difference 
    between grouped query attention and multi head  - for each box we times Embedding Space/ value Heads by softmax which is No tokens * No tokens.
    But this is done for all 14 heads ie query heads.
    """
    # Dimension of value head
    dim_value_head = embedding_dim // value_heads 
    # Computations of (No_tokens, Value Head Dim) * (No_tokens, No_tokens)
    multiplications_heads = no_tokens * no_tokens * dim_value_head
    additions_head = no_tokens * no_tokens * (dim_value_head - 1)

    # Sum over all these heads
    multiplications = multiplications_heads * query_heads
    additions = additions_head * query_heads

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]

    return np.array(results_array)

def concatentation_flops(query_heads):
    """
    In this we calculate the number of flops in concatenating the attention heads together

    This is a simple concatenation ie a memory operation and thus requires no flops in the forward pass
    """
     
    # [additions, multiplications, divisions, exponentiations, square_root]
    return np.array([0, 0, 0, 0, 0])

def residual_connection_flops(no_tokens, embedding_dim):
    """
    In this we calculate the number of flops in the residual connection

    This is applied after the attention mechanism and after the MLP, to carry forward embedding space 
    representation to the next layer.
    """
    additions = no_tokens * embedding_dim

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, 0*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]

    return np.array(results_array)

def rmsnorm_flops(no_tokens, embedding_dim):
    """
    In this we calculate the number of flops in the RSMNorm layer

    This is simply a squared term, summed and then square rooted across the embedding dimension, this is done twice
    after the self attention mechanism and the MLP, this is preformed twice however each time it is the same operation 
    and can be applied twice.
    """
    # Multiple all terms by themselves
    multiplications = no_tokens * embedding_dim 
    # Sum all terms along embedding dimension
    additions =  no_tokens * (embedding_dim - 1)
    # After you sum over all of embedding space you have a seqence of length - no tokens
    divisions = no_tokens # This is for the dividing the constant
    sqrt = no_tokens
    # Then need to divide all terms by the square root
    divisions += no_tokens * embedding_dim
    multiplications += no_tokens * embedding_dim

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, divisions*flops_cost_division, 0*flops_cost_exponentiation, sqrt*flops_cost_sqrt]

    return np.array(results_array)


def mlp_flops(no_tokens, embedding_dim, mlp_hidden_dim):
    """
    This goes up to the dimension of MLP_output_dim (4864) and then back down to the original embedding space

    There are no bias terms in the MLP layers in the Qwen model and this not included in the calculations.
    There are also activaction (applied to the middle - higher dimensionsal layer) but the flops of this are
    included in the SwiGLU function.
    """
    # Projection up to the higher dimension
    multiplications_up = no_tokens *  embedding_dim * mlp_hidden_dim 
    additions_up       = no_tokens * ((embedding_dim -1) * mlp_hidden_dim)
    # Projection down to the original embedding space
    multiplications_down = no_tokens *  embedding_dim * mlp_hidden_dim
    additions_down       = no_tokens * embedding_dim * (mlp_hidden_dim - 1)
    # Sum over all these two layers
    multiplications_overall = multiplications_up + multiplications_down
    additions_overall = additions_up + additions_down
    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions_overall*flops_cost_addition, multiplications_overall*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]
    return np.array(results_array)

def silu_flops(no_tokens, MLP_hidden_dim):
    """
    In this we calculate the number of flops in the SiLU activation function, this is applied in the MLP layer of dimension
    4864.
    
    This activation function is only applied once in the MLP layer and thus the flops are only calculated once. There is some 
    ambiguity as the paper described SWIGLU and not SiLU, however when printing a summary of the model it implies that the 
    Silu activation function is being used
    """
    # Single each of these for each element in the MLP layer across all tokens
    exponentiations = no_tokens * MLP_hidden_dim 
    additions =  no_tokens * MLP_hidden_dim 
    divisions = no_tokens * MLP_hidden_dim 

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, 0*flops_cost_multiplication, divisions*flops_cost_division, exponentiations*flops_cost_exponentiation, 0*flops_cost_sqrt]

    return np.array(results_array)


################ THIS IS WHERE IT ALL GETS AGGREGATED TOGETHER ####################

def forwards_pass_flops(no_tokens):
    """
    This function calculates the total number of flops in the forward pass of the Qwen model

    It uses the following architectures:
    - Positional Encoding to Embedding Space
    - 24 Transformer Layers:
        - RMSNorm
        - Multi-Head Attention (14 Query Heads, 2 Key Heads, 2 Value Heads)
        - Residual Connection
        - RSMNorm
        - MLP (Linear, Activation Function, Linear)
        - Residual Connection
    - RMSNorm
    """
    ###
    num_layers = 24
    embedding_dim = 896
    query_heads = 14
    key_heads = 2
    value_heads = 2
    mlp_hidden_dim = 4864

    ###
    flops_cost_addition = 1
    flops_cost_multiplication = 1 
    flops_cost_division = 1
    flops_cost_exponentiation = 10
    flops_cost_sqrt = 10

    # Initialise the total number of flops for the forward pass
    # [additions, multiplications, divisions, exponentiations, square_root]
    total_flops = np.array([0,0,0,0,0])

    # Compute the number of flops for the positional embedding layer
    total_flops +=  flops_positional_embedding(no_tokens, embedding_dim)

    # This part is for a single later - we will multiply this by the number of layers
    # [additions, multiplications, divisions, exponentiations, square_root]
    single_layer_flops = np.array([0,0,0,0,0])

    single_layer_flops += rmsnorm_flops(no_tokens, embedding_dim)
    # Attention Head
    single_layer_flops += query_head_flops(no_tokens, embedding_dim, query_heads)
    single_layer_flops += key_head_flops(no_tokens, embedding_dim, key_heads)
    single_layer_flops += value_head_flops(no_tokens, embedding_dim, value_heads)
    single_layer_flops += attention_mechanism_flops(no_tokens, embedding_dim, query_heads)
    single_layer_flops += softmax_flops(no_tokens, query_heads)
    single_layer_flops += softmax_values_flops(no_tokens, embedding_dim, value_heads, query_heads)
    single_layer_flops += concatentation_flops(query_heads) # This is zero but included for completeness

    single_layer_flops += residual_connection_flops(no_tokens, embedding_dim)
    single_layer_flops += rmsnorm_flops(no_tokens, embedding_dim)
    # MLP Layer
    single_layer_flops += mlp_flops(no_tokens, embedding_dim, mlp_hidden_dim)
    single_layer_flops += silu_flops(no_tokens, mlp_hidden_dim)

    single_layer_flops += residual_connection_flops(no_tokens, embedding_dim)

    # Across all 24 Transformer layers
    all_layer_flops = single_layer_flops * num_layers
    total_flops += all_layer_flops

    # There is a final RMSNorm layer at the end of the model
    total_flops += rmsnorm_flops(no_tokens, embedding_dim)

    return total_flops, single_layer_flops


### Final equation

def qwen_model_flops(no_tokens, no_batches, num_steps_training):

    forward_pass, single_layer_flops = forwards_pass_flops(no_tokens)
    forward_back_pass = forward_pass * 3

    training_flops = no_batches * num_steps_training * forward_back_pass

    # Header
    headers = ["Additions", "Multiplications", "Divisions", "Exponentiations", "Square Roots"]
    print(f"{'Operation Cost':<20} " + " ".join(f"{h:<18}" for h in headers))

    # Single Layer
    print(f"{'Single Layer':<20} " + " ".join(f"{v:.3g}".ljust(18) for v in single_layer_flops))

    # Forward Pass
    print(f"{'Forward Pass':<20} " + " ".join(f"{v:.3g}".ljust(18) for v in forward_pass))

    # Full Training
    print(f"{'Total Training':<20} " + " ".join(f"{v:.3g}".ljust(18) for v in training_flops))

    # Overall Total
    overall_total = np.sum(training_flops)
    print(f"\n{'Overall Total FLOPs:':<20} {overall_total:.3g}")





# print('The lora adaptation will only need to run once per backpropagation step and hence scale with invrse batch size')

# #  For lora model we have
# # Query heads: 
# lora_q = no_tokens * embedding_dim / q_heads * q_heads * (lora_rank * flops_cost_multiplication + (lora_rank -1) * flops_cost_addition) * num_layers

# # Value heads:
# lora_v = no_tokens * embedding_dim / q_heads * v_heads * (lora_rank * flops_cost_multiplication + (lora_rank -1) * flops_cost_addition) * num_layers

# # Key   heads:
# lora_k = no_tokens * embedding_dim / q_heads * k_heads * (lora_rank * flops_cost_multiplication + (lora_rank -1) * flops_cost_addition) * num_layers

# # lora total cost
# lora_total = lora_q + lora_v + lora_k
# print('')
# print(f"Total cost of Lora: {lora_total:.2e}")