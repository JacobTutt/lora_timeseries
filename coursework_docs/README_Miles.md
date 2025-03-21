# M2 Coursework: LoRA Fine-Tuning for Time Series Forecasting

Please open the file `main.pdf` for the coursework instructions. Good luck and have fun!

## Clarifications

### Q: Are the differential equations used to generate the Lotka-Volterra data the standard equations or modified versions? Are the original parameters available and should they be used as model inputs?

A:

The specific differential equations used to generate the data are not relevant. You should treat the dataset as observations you have been given and are now trying to model with a time series forecast approach.

The idea is that the LLM will pick up on the patterns from its input and be able to infer any relevant parameters in the underlying dynamics, similar to the approach in the LLMTIME paper.

This is similar to how LLMs work in practice. They predict the next token based on the previous sequence of tokens. They don't need structured inputs describing if the user is happy or sad. They just infer it all from the context.


### Q: Do we need to track FLOPs inside the model, every single operation?

This is probably overkill. All I was thinking of was something like the following:

```python
def flops_mlp(num_layers, num_hidden, input_dim):
    """Compute the FLOPs for a given MLP"""
    flops = 0
    layer_dims = [input_dim] + [num_hidden] * num_layers

    for i in range(num_layers):
        # matrix multiplication:
        flops += (2 * layer_dims[i] - 1) * layer_dims[i + 1]
        if i < num_layers - 1:
            # ReLU activation (except last layer)
            flops += layer_dims[i + 1]

    return flops

def flops_forward_pass(batch_size, num_layers, num_hidden, input_dim):
    """Compute the FLOPs for a forward pass in the training loop."""
    return batch_size * flops_mlp(num_layers, num_hidden, input_dim)

def flops_forward_and_back(batch_size, num_layers, num_hidden, input_dim):
    # (We simplify things and assume backward = 2x forward, as per instruction in coursework)
    return 3 * flops_forward_pass(batch_size, num_layers, num_hidden, input_dim)

def total_flops(num_steps_training, batch_size, num_layers, num_hidden, input_dim):
    """Total FLOPs for the experiment"""
    return num_steps_training * flops_forward_and_back(batch_size, num_layers, num_hidden, input_dim)
```

Which computes the FLOPs for a given MLP (this example assumes ReLU; you would need to write yours differently).

So, if we were training a 3-layer MLP with 256 hidden units, ReLU activations, input dimension 5, batch size 32, and 10,000 calls to `.step()`, the total FLOPs would be: `total_flops(10000, 32, 3, 256, 5)` (although note that we left out the parameter update steps.)

So, you could think about writing a `total_flops` for your experiments given the hyperparameters you define.
Consider writing it in a modular way so you can just re-use function calls for repeated blocks.

Then you could just compute this for a given experiment, and put that number in your table.

### Q: Should $\alpha$ in the preprocessing scheme be chosen for each trajectory, or once for the entire dataset?

For the context of this paper, it should be chosen once for the entire dataset. In other words, you would only have a single scalar $\alpha$ in total.

However, if you have already chosen to do it with $\alpha$ chosen per trajectory, that is also fine. Just say which one you went with.
