import math
import numpy as np

# GLOBAL VARIABLES - these are also applied in the functions
flops_cost_addition = 1
flops_cost_multiplication = 1 
flops_cost_division = 1
flops_cost_exponentiation = 10
flops_cost_sqrt = 10

budget = 1e17


# I first outline the flops in each section of the QWEN (+ Lora) model
# Later functions pool all of these functions together and calculate the total flops for the model
# Each function returns the overall cost each operation ie [additions, multiplications, divisions, exponentiations, square_root] in flops

# BEFORE THE TRANSFORMER LAYERS

def convert_tokens_to_embeddings():
    """
    Determines the number of flops required to convert tokens from hot encoding vectors to embedding vectors.

    This is treated as a memory task (/mapping) operation during the forward pass and this does not add to our overall 
    flops budget.
    """
    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [0, 0, 0, 0, 0]
    return np.array(results_array)

# def rotary_positional_embedding_flops(no_tokens, embedding_dim, query_heads):
#     """
#     Compute the flops for a single data element in a positional embedding layer.

#     The 
#     This is a simple addition operation at the beginning of the embegging space which adds a positional embedding 
#     to every value in th embedding input space (no. tokens x dimension fo embedding space)
#     """

#     additions = no_tokens * embedding_dim

#     # [additions, multiplications, divisions, exponentiations, square_root]
#     results_array = [additions*flops_cost_addition, 0*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]
#     return np.array(results_array)


### TRANSFORMER LAYERS
# Note we do not use traditional sinosodial positional embeddings but instead use rotary positional embeddings which is added to the 
# inputs to the query and value heads. This is treated as a matrix multiplication on the global
def rotary_positional_embedding_flops(no_tokens, embedding_dim):
    """
    Compute the flops for a each rotary positional embedding addition.

    This is applied within each layer in the Qwen model and is also applied to the input of both the query and value 
    heads and this applied twice per layers.

    The calculation of flops assumes that we are not required to calculate the flops of the rotation matrix itself however
    we need to preform matrix multiplication to apply this rotation. This calculation assumes we apply the rotation to the 
    global query and values across the heads although the compute is the same if applied to each head individually.

    Rotary positional embeddings is type of positional encoding which with a rotation matrix and naturally incorporates
    explicit relative position dependency in self-attention formulation.
    """

    multiplications = no_tokens * embedding_dim * embedding_dim
    additions = no_tokens * (embedding_dim - 1) * embedding_dim

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]
    return np.array(results_array)

### Self Attention Mechanism

def query_head_flops(no_tokens, embedding_dim, query_heads):
    """
    Calculate the number of flops in generating the query matrix for the attention mechanism across all heads.

    In Qwen the query projection includes biases being added and these are accounted for.

    In the Qwen model we have 14 of these heads so the dimemnsionality of the embedding space is split across
    these heads. However the number of flops in the same if calculated for a single matrix multiplication on the 
    global weights across all heads or the weights for each head individually.

    This calculation is broken down explicitly for each head for clarity.
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
    Calculate the number of flops in generating the key matrix for the attention mechanism

    In the Qwen model we have 2 keys heads. You can think of this as either a simple multi head attention where the embedding space
    is split over the total 14 heads or group query attention where the embedding space is split over the 2 key heads. Either senario
    the number of flops is the same as it is calculated across all heads. As you simply divide the embedding space by the number of heads 
    and then re-multiply it up for number of heads. 

    This calculation is broken down explicitly for each head for clarity.
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
    Calculate the number of flops in generating the value matrix for the attention mechanism

    In the Qwen model we have 2 keys heads. You can think of this as either a simple multi head attention where the embedding space
    is split over the total 14 heads or group query attention where the embedding space is split over the 2 key heads. Either senario
    the number of flops is the same as it is calculated across all heads. As you simply divide the embedding space by the number of heads 
    and then re-multiply it up for number of heads. 

    This calculation is broken down explicitly for each head for clarity.
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
    Calculate the number of flops in the attention mechanism to work out relations in queries and keys. 

    This is preformed for each head - there is some ambiguity in the qwen model as this it uses Grouped Query Attention, 
    the follow calculations are based of the mathematics of Multi-Head Attention with 14 query heads, and thus 14 seperate 
    attentions matrices are calculated. 

    This also accounts for the normalisation of the attention matrix by dividing by the square root of the dimensionality of the
    query and key heads (a constant)
    
    This is broken down for each head for clarity.
    """
    
    # The number of dimensions per head - 
    dim_query_head = embedding_dim // query_heads  # Dimension per query head - will be 64 but laid out explicitly
    # Multiplications per head
    multiplications_head = no_tokens * no_tokens * dim_query_head
    # Additions per head
    additions_head = no_tokens * no_tokens * (dim_query_head - 1) 
    # The whole matrix is also normalised by dividing by the dimensionality of the query and keys
    square_root = 1 # Can be reused for all heads and stored in memeory - this is one operation in the flops
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
    Calculate the number of flops in the softmax operation in the attention mechanism for each head and combine this across 
    all heads.

    We are treating this as multi head attention with 14 heads, and thus the softmax is calculated for each and multiplied by 14.

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


def softmax_values_flops(no_tokens, embedding_dim, query_heads):
    """
    Calculate multiplting the values projection with the attention individually and then across all heads
    
    This is one of the places were the difference between grouped query attention and multi head attention is noticable 
    
    For each head we times Embedding Space/  no. heads by softmax which is No tokens * No tokens but this is done for all 
    14 heads ie query heads.
    """
    # Dimension of value head
    dim_value_head = embedding_dim // query_heads 
    # Computations of (No_tokens, Value Head Dim) * (No_tokens, No_tokens)
    multiplications_heads = no_tokens * no_tokens * dim_value_head
    additions_head = no_tokens  * dim_value_head * (no_tokens-1)

    # Sum over all these heads
    multiplications = multiplications_heads * query_heads
    additions = additions_head * query_heads

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]

    return np.array(results_array)


def concatentation_flops(query_heads):
    """
    In this we calculate the number of flops in concatenating the attention heads together into a single matrix.

    This is a simple concatenation ie a memory operation and thus requires no flops in the forward pass
    """
     
    # [additions, multiplications, divisions, exponentiations, square_root]
    return np.array([0, 0, 0, 0, 0])

### Linear Mixing, Residual Connections and RMSNorm layers

def linear_mixing_flops(no_tokens, embedding_dim):
    """
    In this we calculate the number of flops in the linear mixing of the attention heads.

    After the concatenation of the multiple self attention heads we then apply a linear mixing to the concatenated heads, 
    no bias is applied simply a multiplication of a (embedding_dim x embedding_dim) matrix to the concatenated heads.
    """
    multiplications = no_tokens * embedding_dim * embedding_dim
    additions = no_tokens * (embedding_dim - 1) * embedding_dim

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]
    return np.array(results_array)


def residual_connection_flops(no_tokens, embedding_dim):
    """
    In this we calculate the number of flops in the residual connection, which is a simple addition operation across the 
    embedding and tokens space.

    This is applied after the attention mechanism and after the MLP, to carry forward embedding space representation to the 
    next layer. ie twice in each later
    """
    additions = no_tokens * embedding_dim

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, 0*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]

    return np.array(results_array)


def rmsnorm_flops(no_tokens, embedding_dim):
    """
    In this we calculate the number of flops in the RSMNorm layer, which normalised the inputs across the embedding space. 

    This is simply a squared term, summed and then square rooted across the embedding dimension, this is done twice
    after the self attention mechanism and the MLP. As the dimensionality is the same in each senario this is defined once but
    applied twice.
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


### Multi-Layer Perceptron with SwiGLU Activation Function

def mlp_flops(no_tokens, embedding_dim, mlp_hidden_dim):
    """
    This goes up to the dimension of MLP_output_dim (4864) and then back down to the original embedding space.

    There is also an addition gate projection up, this can either be consider part of the swiglu of the mlp, 
    for this case it is calculated in the mlp flops and interpreted as part of the MLP.
    The swiGLU activaction is applied in the middle - higher dimensionsal layer) but the flops of this are included 
    in the SwiGLU function.

    There are no bias terms in any of the MLP layers in the Qwen model and thus not included in the calculations.
    """
    # Projection up to the higher dimension - nevauis
    multiplications_up = no_tokens *  embedding_dim * mlp_hidden_dim 
    additions_up       = no_tokens * ((embedding_dim -1) * mlp_hidden_dim)
    # gate projection, this can either be considered as part of the swiglu or the mlp, it is catagories as the mlp
    multiplications_gate = no_tokens *  embedding_dim * mlp_hidden_dim 
    additions_gate       = no_tokens * ((embedding_dim -1) * mlp_hidden_dim)
    # Projection down to the original embedding space
    multiplications_down = no_tokens *  embedding_dim * mlp_hidden_dim
    additions_down       = no_tokens * embedding_dim * (mlp_hidden_dim - 1)
    # Sum over all these two layers
    multiplications_overall = multiplications_up + multiplications_down + multiplications_gate
    additions_overall = additions_up + additions_down + additions_gate
    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions_overall*flops_cost_addition, multiplications_overall*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]
    return np.array(results_array)


def swiglu_flops(no_tokens, mlp_hidden_dim):
    """
    In this we calculate the number of flops in the SwiGLU activation function, this is applied in the MLP layer of dimension
    4864.

    The projection cost of this additional dimension is considered by the mlp function, This function is accounts for appling the swish
    activation function with the mlp up and then a multiplication of the projection up.

    swiGLU(x, W, V, b, c, β) = Swish_β(xW + b) x (xV + c) (element wise multiplication)
    
    This activation function is only applied once in the MLP layer and thus the flops are only calculated once. 

    Assume the negation is zero cost in the activation function.
    """
    # Single each of these for each element in the MLP layer across all tokens
    # This is the silu part of the swiglu
    exponentiations_silu = no_tokens * mlp_hidden_dim 
    additions_silu =  no_tokens * mlp_hidden_dim 
    divisions_silu = no_tokens * mlp_hidden_dim 

    # We then times this silu results ( no_tokens x mlp_hidden_dim) by the gate projection up (no_tokens x mlp_hidden_dim)
    # In a element wise multiplication
    multiplications_down = no_tokens * mlp_hidden_dim

    multiplications = multiplications_down
    additions = additions_silu 
    divisions = divisions_silu
    exponentiations = exponentiations_silu

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, divisions*flops_cost_division, exponentiations*flops_cost_exponentiation, 0*flops_cost_sqrt]

    return np.array(results_array)

### Final Layers

def embed_vocab_linear(no_tokens, embedding_dim, vocabuluary_dimension):
    """
    In this we calculate the number of flops in the final linear layer to go from the embedding space to the
    vocabulary space, with is simply a linear layer with weights and bias.

    Start from 4864 and go to 50257
    """

    # Cost of matrix multiplication
    multiplications = no_tokens * embedding_dim * vocabuluary_dimension
    addition_matrix = no_tokens * (embedding_dim - 1) * vocabuluary_dimension

    # Cost of bias addition
    addition_bias = no_tokens * vocabuluary_dimension

    # Overall additions
    additions = addition_matrix + addition_bias

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]
    return np.array(results_array)

### Final Softmax Layer only applied in probabilistic sampling

def final_softmax_flops(vocabuluary_dimension):
    """
    In this we calculate the number of flops in the final softmax layer over the vocabulary space, this is used in probabilistic 
    sampling to reweight all of the values in the vocabulary space.

    This is not required if using greedy sampling as this simply takes the maximum value in the vocabulary space, 
    and thus normalisation into probabilities is not required.

    It is simply applied on a matrix of the dimensions (no_tokens, vocabuluary_dimension)
    """
    # Calculate each - sum and then normalise through division
    exponentiations = vocabuluary_dimension 
    additions = (vocabuluary_dimension - 1)
    divisions = vocabuluary_dimension

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, 0*flops_cost_multiplication, divisions*flops_cost_division, exponentiations*flops_cost_exponentiation, 0*flops_cost_sqrt]
    return np.array(results_array)

### LoRA Rank Adaptation

def lora_rank_flops(no_tokens, lora_ranks, embedding_dim): # might add query_heads in here if doing it individually
    """
    In this we calculate the number of flops in the LoRA rank adaptation to the attention mechanism

    This is applied to the query projection and value projection layers of the attention mechanism and is a seperate channel of 
    matrix multiplication (2 steps) and then added to the original proection matrices. 

    This is treating the query and value heads as a global matricies across the mutliple self attention heads and thus 
    is applied to the full matrix once.

    The input is N_tokens x Embedding_Dimensions
    The dimension of the query and value heads is No_Tokens * Embedding_Dimensions

    We multiple these together to get a No_Tokens * Embedding_Dimensions matrix.
    We then add these to the original global query and value heads to get the new updated versions. 

    So I will apply this flops twice per tranformer layer, once for the query heads and once for the value heads.
    """
    # # The number of dimensions per head
    # dim_query_head = embedding_dim // query_heads  # Dimension per query head - will be 64 but laid out explicitly

    # Downwards projection - this is (N_tokens,Embed dim) * (Embed, LoRA_Ranks) ending is (N_tokens, LoRA_Ranks)
    multiplications_down = no_tokens * embedding_dim * lora_ranks
    additions_down = no_tokens * (embedding_dim - 1) * lora_ranks

    # Upwards projection - this is (N_tokens, LoRA_Ranks) * (LoRA_Ranks, Embed dim) ending is (N_tokens, Embed/Heads)
    multiplications_up = no_tokens * lora_ranks * embedding_dim #dim_query_head
    additions_up = no_tokens * (lora_ranks - 1) * embedding_dim #dim_query_head

    # There is an alpha over r multiplcation
    multiplications_scalar = no_tokens * embedding_dim

    # There is now a cost associated with adding this resulting matrix to the original query head
    additions_lora_orig = no_tokens * embedding_dim

    multiplications = multiplications_down + multiplications_up + multiplications_scalar
    additions = additions_down + additions_up + additions_lora_orig

    # Sum over all these heads
    # additions = additions_head * query_heads # (14)
    # multiplications = multiplications_head * query_heads # (14)

    # [additions, multiplications, divisions, exponentiations, square_root]
    results_array = [additions*flops_cost_addition, multiplications*flops_cost_multiplication, 0*flops_cost_division, 0*flops_cost_exponentiation, 0*flops_cost_sqrt]

    return np.array(results_array)


################ THIS IS WHERE IT ALL GETS AGGREGATED TOGETHER ####################
###################################################################################


def forwards_pass_flops(no_tokens, lora_ranks, print_summary = False):
    """
    This function calculates the total number of flops in the forward pass of the Qwen model, and is able to account for additional
    flops if the LoRA rank adaptation is applied to the model.

    It uses the following architectures:
    - Token to Embedding Layer (no compute)
    - 24 Transformer Layers:
            - RMSNorm
            - Rotary Positional Embedding (Query and Value Heads)
            - Grouped Query Attention/ Multi Head Attention (14 Query Heads, 2 Key Heads, 2 Value Heads)
                    - Query, Key, Value Heads + LoRA Rank Adaptation if included
                    - Attention Mechanism
                    - Softmax
                    - Self Attention Mechanism (Values * Softmax)
            - Residual Connection
            - RSMNorm
            - MLP (Projection Up, SwiGLU Activation Function, Projection Down)
            - Residual Connection
    - RMSNorm
    - Embedding to Vocabulary Linear Layer


    Parameters
    ----------
    no_tokens : int
        The number of tokens in the input sequence.
    lora_ranks : int
        The rank of the LoRA decomposition. If 0, LoRA is not applied.
    print_summary : bool, optional
        If True, prints a detailed breakdown of FLOPs across different model components.

    Returns
    -------
    total_forward_pass : float
        Total number of FLOPs (summed across all operation types) in the full forward pass.
    total_lora : float
        Total number of FLOPs attributed specifically to LoRA across all transformer layers.
    """
    ###
    num_layers = 24
    embedding_dim = 896
    query_heads = 14
    key_heads = 2
    value_heads = 2
    mlp_hidden_dim = 4864
    vocabuluary_dimension = 151936

    ###
    # Initialise the total number of flops for the forward pass
    # [additions, multiplications, divisions, exponentiations, square_root]
    total_flops = np.array([0,0,0,0,0])
    single_layer_flops = np.array([0,0,0,0,0])
    single_layer_lora_flops = np.array([0,0,0,0,0])

    # Compute the number of flops for the conversion of tokens to embeddings 
    total_flops += convert_tokens_to_embeddings() # This is zero as a memory operation but included for completeness

    # # Compute the number of flops for the positional embedding layer
    # total_flops +=  flops_positional_embedding(no_tokens, embedding_dim)

    # This part is for a single later - we will multiply this by the number of layers
    # [additions, multiplications, divisions, exponentiations, square_root]

    single_layer_flops += rmsnorm_flops(no_tokens, embedding_dim)

    # Multi Head Attention - Calculated across all 14 heads simultaneously
    single_layer_flops += query_head_flops(no_tokens, embedding_dim, query_heads)
    single_layer_flops += rotary_positional_embedding_flops(no_tokens, embedding_dim) # Rotary Positional Embedding for Query Heads

    single_layer_flops += key_head_flops(no_tokens, embedding_dim, key_heads)
    single_layer_flops += rotary_positional_embedding_flops(no_tokens, embedding_dim) # Rotary Positional Embedding for Key Heads

    single_layer_flops += value_head_flops(no_tokens, embedding_dim, value_heads)

    single_layer_flops += attention_mechanism_flops(no_tokens, embedding_dim, query_heads)
    single_layer_flops += softmax_flops(no_tokens, query_heads)

    single_layer_flops += softmax_values_flops(no_tokens, embedding_dim, query_heads)
    single_layer_flops += concatentation_flops(query_heads) # This is zero but included for completeness

    # Linear Mixing
    single_layer_flops += linear_mixing_flops(no_tokens, embedding_dim)

    single_layer_flops += residual_connection_flops(no_tokens, embedding_dim)
    single_layer_flops += rmsnorm_flops(no_tokens, embedding_dim)
    # MLP Layer
    single_layer_flops += mlp_flops(no_tokens, embedding_dim, mlp_hidden_dim)
    single_layer_flops += swiglu_flops(no_tokens, mlp_hidden_dim)
    single_layer_flops += residual_connection_flops(no_tokens, embedding_dim)
    # If lora has been applied to the model we must also calculate the flops for this

    if lora_ranks != 0:
        single_layer_flops += 2 * lora_rank_flops(no_tokens, lora_ranks, embedding_dim) # Done for query matrix and value matrix hence times two
         # This is simply done to document the lora flops
        single_layer_lora_flops +=  2 * lora_rank_flops(no_tokens, lora_ranks, embedding_dim)

    # Across all 24 Transformer layers
    all_layer_flops = single_layer_flops * num_layers
    # Add Transformer layers to total flops
    total_flops += all_layer_flops

    # There is a final RMSNorm layer at the end of the model
    total_flops += rmsnorm_flops(no_tokens, embedding_dim)
    # Then projected up to the vocab space
    total_flops += embed_vocab_linear(no_tokens, embedding_dim, vocabuluary_dimension)
    # # Finally a softmax over the vocab space for probabilties
    # total_flops += final_softmax(no_tokens, vocabuluary_dimension)

    # Compute totals for each row
    single_layer_total = np.sum(single_layer_flops)
    single_layer_lora_total = np.sum(single_layer_lora_flops)
    total_forward_pass = np.sum(total_flops)

    if print_summary:
        headers = ["Additions", "Multiplications", "Divisions", "Exponentiations", "Square Roots", "Total"]
        print(f"{'Operation Cost':<35}" + " ".join(f"{h:<18}" for h in headers))
        

        # Print each row including the total sum
        print(f"{'Single Transformer Layer':<35}" + " ".join(f"{v:.3g}".ljust(18) for v in single_layer_flops) + f"{single_layer_total:.3g}".ljust(18))
        print(f"{'LoRA in Single Layer':<35}" + " ".join(f"{v:.3g}".ljust(18) for v in single_layer_lora_flops) + f"{single_layer_lora_total:.3g}".ljust(18))
        print(f"{'Full Forward Pass':<35}" + " ".join(f"{v:.3g}".ljust(18) for v in total_flops) + f"{total_forward_pass:.3g}".ljust(18))
        
        # Overall total across all FLOPs
        print(f"\n{'Overall Total FLOPs:':<35} {total_forward_pass:.5g}")
        # print percentage of flops budget
        print(f"{'Percentage of Total FLOPs Budget:':<35} {total_forward_pass / budget *100:.5g} %")

    total_lora = single_layer_lora_total * num_layers

    return total_forward_pass, total_lora


def model_generation_flops(tokens_given, tokens_generated, lora_ranks, randomness = False, print_summary = True):
    total_flops = 0
    """
    Calculates the total number of FLOPs required to generate a sequence of tokens using the Qwen model,
    with optional LoRA adaptation and sampling strategy.

    This function models autoregressive generation, where each token is generated one at a time by running 
    a full forward pass through the model, with the context length increasing at each step 
    (i.e., generating token i requires a forward pass with `tokens_given + i` input tokens).

    If `randomness` is enabled, the model uses probabilistic sampling over the vocabulary and applies an additional 
    softmax operation at the final layer for each generated token. If disabled, the model uses greedy decoding 
    (argmax) and skips the final softmax, reducing computation.

    Parameters
    ----------
    tokens_given : int
        The number of context tokens provided at the start of generation.
    tokens_generated : int
        The number of tokens to generate (each token requires a forward pass).
    lora_ranks : int
        Rank of the LoRA adaptation. If set to 0, LoRA is not included in the calculation.
    randomness : bool, optional
        If True, adds FLOPs for a softmax over the vocabulary at each generation step. Defaults to False.
    summary : bool, optional
        Reserved for optional detailed summary output (currently unused). Defaults to False.

    Returns
        -------
        total_flops : float
            Total number of FLOPs for generating the full sequence (summed across all operations and steps).
        total_lora_flops : float
            Total number of FLOPs that specifically result from the LoRA adaptation across all generation steps.
    """
    vocabuluary_dimension = 151936
    total_flops = 0
    total_lora_flops = 0

    # Each token generated requires a forward pass of the model so 
    for i in range(tokens_generated):
        generation_flops, lora_flops = forwards_pass_flops(tokens_given + i, lora_ranks, print_summary = False)
        total_flops += generation_flops
        total_lora_flops += lora_flops

    # If randomness is used, we require than an additional softmax is applied to the final output vocabulaury
    # so it can then be sampled from the probability distribution of vocabuluary space.
    if randomness:
        total_flops += tokens_generated * np.sum(final_softmax_flops(vocabuluary_dimension))

    if print_summary:
        print (f"Total FLOPs for generating {tokens_generated} tokens: {total_flops:.5g}")
        print (f"Total FLOPs from LoRA adaptation: {total_lora_flops:.5g}")
            # print percentage of flops budget
        print(f"{'Percentage of Total FLOPs Budget:':<35} {total_flops / budget *100:.5g} %")
    
    return total_flops, total_lora_flops



### Final equation

def model_training_flops(no_tokens, lora_ranks, batch_size, num_steps_training, print_summary = True):
    """
    Calculates the total number of FLOPs required to train the Qwen model, including LoRA adaptation if specified.

    The function assumes a standard training loop where each step involves:
    - A forward pass
    - A backward pass (with a computation cost of a 2x forward pass)

    This is then repeated for each batch in the training set and for each training step. ie multiplied by.

    Note:
    -----
    Unlike generation, training does not require applying a softmax over the vocabulary.
    This is because training is typically based on raw logits and uses loss functions 
    like cross-entropy that apply softmax internally as part of the loss computation.

    Parameters
    ----------
    no_tokens : int
        Number of tokens in each input sample.
    lora_ranks : int
        Rank of the LoRA decomposition. If set to 0, LoRA is not applied.
    batch_size : int
        Number of batches per epoch (or per training session).
    num_steps_training : int
        Number of training steps (or gradient updates).

    Returns
    -------
    training_flops : np.ndarray
        The total training FLOPs across all steps and batches.

    training_lora_flops : np.ndarray
        The total FLOPs attributed specifically to LoRA components (if applied), across all training steps.
    """
    # For training we need to run the forward pass and the backward over 
    forward_pass, lora_pass = forwards_pass_flops(no_tokens, lora_ranks, print_summary = False)

    forward_back_pass = forward_pass * 3
    lora_forward_back_pass = lora_pass * 3

    training_flops = batch_size * num_steps_training * forward_back_pass
    training_lora_flops = batch_size * num_steps_training * lora_forward_back_pass

    if print_summary:
        print(f"Total FLOPs for training: {training_flops:.5g}")
        print (f"Total FLOPs from LoRA adaptation: {training_lora_flops:.5g}")
        # print percentage of flops budget
        print(f"{'Percentage of Total FLOPs Budget:':<35} {training_flops / budget *100:.5g} %")

    return training_flops, training_lora_flops


def model_evaluation_flops(no_tokens, lora_ranks, batch_size, print_summary = True):
    """
    Calculates the total number of FLOPs required to evaluate the Qwen model over a dataset,
    optionally including LoRA adaptation overhead.

    Evaluation consists of a single forward pass through the model per batch, followed by loss 
    computation (typically cross-entropy), which operates directly on the vocabulary logits 
    and the ground truth token targets.

    Unlike generation, evaluation does not apply a softmax over the vocabulary distribution
    during inference, as the loss function (e.g. cross-entropy) internally handles this step.

    Parameters
    ----------
    no_tokens : int
        Number of tokens in each input sample.
    lora_ranks : int
        Rank of the LoRA decomposition. If set to 0, LoRA is not applied.
    no_batches : int
        Number of batches in the evaluation dataset.

    Returns
    -------
    evaluation_flops : np.ndarray
        The total evaluation FLOPs across all batches

    evaluation_lora_flops : np.ndarray
        The total FLOPs specifically attributed to LoRA components (if applied), across all batches.
    """
    # For model evaluation we simply need to run the forward pass over the model and then calculate the loss using the vocabuluary space
    # logits and the target labels + 1.

    forward_pass, single_lora_flops = forwards_pass_flops(no_tokens, lora_ranks, print_summary = False)

    evaluation_flops = batch_size * forward_pass
    evaluation_lora_flops = batch_size * single_lora_flops

    if print_summary:
        print (f"Total FLOPs for evaluation: {evaluation_flops:.5g}")
        print (f"Total FLOPs from LoRA adaptation: {evaluation_lora_flops:.5g}")
        # print percentage of flops budget
        print(f"{'Percentage of Total FLOPs Budget:':<35} {evaluation_flops / budget * 100:.5g} %")

    return evaluation_flops, evaluation_lora_flops
    


