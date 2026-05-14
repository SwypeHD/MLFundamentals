# Future Neural Network Optimizations

Roadmap of improvements for `neuralnetwork.py` to make it actually trainable on MNIST in a reasonable amount of time. Ordered roughly by impact-per-effort.

## 1. Cross-entropy loss + softmax output

**Why:** MSE loss combined with a sigmoid output saturates badly. When the output is wrong and stuck near 0 or 1, σ'(z) ≈ 0, so `δ^L = (a^L - y) ⊙ σ'(z^L)` is near zero — the network has no gradient signal to correct itself with. This is the vanishing-gradient problem hitting at the output layer specifically.

**How:** For multi-class classification:
- Output layer uses softmax instead of sigmoid: `a^L_i = exp(z^L_i) / Σ_j exp(z^L_j)`. Output is a probability distribution.
- Cost becomes cross-entropy: `C = -Σ_i y_i · log(a^L_i)` where y is one-hot.
- The combined softmax + cross-entropy derivative simplifies beautifully:
  ```
  δ^L = a^L - y
  ```
  No σ' factor. The error signal stays strong even when the network is very wrong.

**Caveat:** Numerical stability — naive softmax overflows for large z. Use the standard trick: subtract `max(z)` before exponentiating. `exp(z - max(z))` has the same softmax output and never overflows.

## 2. Better weight initialization (Xavier / Glorot)

**Why:** Current code uses `random()` which draws from `[0, 1)` — all positive. Combined with all-positive inputs (random or normalized pixels), pre-activations `z = Wx + b` skew large and positive, saturating sigmoid from iteration 1. Even with cross-entropy fixing the output layer, hidden layers still suffer.

**How:** Use a zero-mean distribution scaled by fan-in. For sigmoid/tanh activations:
```python
W = np.random.randn(out_dim, in_dim) * np.sqrt(1 / in_dim)
```
For ReLU activations, use He init instead:
```python
W = np.random.randn(out_dim, in_dim) * np.sqrt(2 / in_dim)
```
Biases can stay at zero (or small random values).

The scaling preserves the variance of activations across layers, so signal doesn't blow up or vanish as it propagates.

## 3. ReLU activation in hidden layers

**Why:** Sigmoid saturates at both ends and has a tiny derivative everywhere — max σ'(z) is 0.25, and it drops fast. Through 3 hidden layers, the gradient gets multiplied by σ' three times, so deep gradients can shrink by a factor of 64+ even before considering weight magnitudes. ReLU's derivative is 1 for positive inputs and 0 for negative — no shrinkage on the active path.

**How:** Replace sigmoid only on hidden layers, keep softmax/sigmoid on the output:
```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

Backprop changes: wherever you currently compute `σ'(z)` on a hidden layer's z, use `relu_derivative(z)` instead. The output layer keeps its own activation/derivative (softmax for cross-entropy, or sigmoid if staying with MSE).

**Caveat:** Dead ReLUs — neurons stuck with negative pre-activations get zero gradient forever. He init plus low learning rates mitigate this. If it becomes an issue, switch to Leaky ReLU: `max(0.01·z, z)`.

## 4. Mini-batch SGD with vectorized batch math

**Why:** Currently the loop processes one input over and over (same `L1_activations` forever). For real training:
- Need to iterate over the actual MNIST dataset.
- Per-sample updates (true SGD with batch size 1) are noisy and slow per epoch.
- Full-batch GD on 60k MNIST images is expensive per step.
- Mini-batches (32–256 samples) are the standard tradeoff: stable gradient estimates with good GPU/CPU utilization.

**How:**
- Load MNIST. Either `keras.datasets.mnist`, `torchvision.datasets.MNIST`, or download the raw IDX files. Normalize pixels to `[0, 1]` (divide by 255).
- Reshape activations to be batched: `L1` becomes shape `(batch_size, 784)` instead of `(784,)`. Every other layer follows.
- Update all the matmul operations to handle the batch dimension:
  - Forward: `z = X @ W.T + b` (shape `(batch, out_dim)`)
  - Backprop: deltas become `(batch, layer_dim)` instead of `(layer_dim,)`.
  - Weight gradients: `dW = delta.T @ a_prev / batch_size` (averages the gradient over the batch).
  - Bias gradients: `db = delta.mean(axis=0)`.
- Wrap the iteration loop:
  ```python
  for epoch in range(num_epochs):
      shuffle(training_data)
      for batch in batches(training_data, batch_size):
          forward(batch)
          backward(batch)
          update_params()
  ```

This is also where vectorization pays off — the per-sample loop disappears and NumPy's matmul handles the entire batch at once.

## Order of operations

If implementing one at a time, sensible order:

1. **Cross-entropy + softmax first.** Biggest single win on the existing architecture. Will make training feasible even without other changes.
2. **Better init.** Free improvement, two-line change, prevents wasted iterations getting out of the saturation regime.
3. **Mini-batch SGD with real MNIST data.** Without real data, the rest doesn't matter. Worth doing once items 1-2 are stable on the single-example overfit case.
4. **ReLU.** Largest code change in terms of touching the math. Worth waiting until after the rest is debugged so you know any new issues come from the ReLU swap, not from interactions with broken cross-entropy or bad init.

## Other things to consider later

- **Learning rate schedules.** Constant `learning_rate = 0.01` works fine to start but most setups decay it over time (e.g., halve every N epochs).
- **Momentum / Adam.** Optimizers that accumulate gradient history converge faster than vanilla SGD on most problems.
- **Validation set + early stopping.** Track loss on held-out data to detect overfitting and stop training when it starts to climb.
- **Regularization.** L2 weight decay or dropout, once you're at the point of overfitting the training set.

These are nice-to-haves rather than required for a working MNIST classifier — items 1-4 above are enough to get to ~97% test accuracy.
