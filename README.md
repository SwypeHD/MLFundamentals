# ML Fundamentals

Personal practice repo for implementing core machine learning concepts from scratch in Python and NumPy. The goal is to build intuition for the underlying math by deriving and writing the algorithms by hand, without relying on libraries like PyTorch or scikit-learn (For some projects). Others are more high-level projects such as MCP and agentic development.

## Contents

### `regression.py` — Polynomial Regression via Gradient Descent

Fits weights for a polynomial basis using batch gradient descent on mean-squared error. The default true function is a random cubic `ax³ + bx² + cx + d`; a `sin(x)` variant is included to demonstrate how the same polynomial basis can approximate non-polynomial targets (a Maclaurin-style fit).

Key details:

- Inputs are z-score normalized so that higher-degree basis values don't explode out of scope.
- Compares the iterative GD solution against the closed-form least-squares solution `w = (IᵀI)⁻¹ Iᵀy` as a correctness check.
- The feature map `I` is built explicitly from the basis functions applied to each input.

### `neuralnetwork.py` — Multilayer Perceptron from Scratch

A 3-hidden-layer feedforward neural network with sigmoid activations and MSE loss, sized for MNIST digit classification (784 → 16 → 16 → 10). The forward pass and backpropagation are hand-derived and implemented with NumPy primitives — no autograd.

What's implemented:

- Forward pass with interleaved pre-activation (`z`) and activation (`σ(z)`) computations.
- Output-layer error signal `δ^L = (a^L - y) ⊙ σ'(z^L)`.
- Backprop through hidden layers via `δ^l = ((W^(l+1))ᵀ δ^(l+1)) ⊙ σ'(z^l)`.
- Weight gradients as outer products `∂C/∂W^l = δ^l ⊗ (a^(l-1))ᵀ`.
- Bias gradients `∂C/∂b^l = δ^l`.

Current state: overfits a single random input to a one-hot target as a correctness check that the gradients actually decrease the loss. Real MNIST data loading and minibatch training are not yet wired up.

## Running

```bash
pip install numpy
python regression.py
python neuralnetwork.py
```

## Notes

The focus is on derivation and implementation clarity over performance or generality. Code uses explicit NumPy operations rather than abstracted layer/optimizer classes so the math stays visible.
