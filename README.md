# ML Fundamentals

Personal practice repo for learning and experimenting with machine learning concepts in Python and NumPy. This is not a polished library — it's a workspace for building intuition by working through the math and code myself.

Some components are written entirely by me from scratch. Others involved AI assistance at specific points (noted below). The goal is learning, not exclusivity of authorship.

## Contents

### [regression.py](regression.py) — Polynomial Regression via Gradient Descent

Written by me. Fits weights for a polynomial basis using batch gradient descent on mean-squared error. The default true function is a random cubic `ax³ + bx² + cx + d`; a `sin(x)` variant demonstrates how the same polynomial basis approximates non-polynomial targets (a Maclaurin-style fit).

Key details:

- Inputs are z-score normalized so higher-degree basis values don't explode — the normalization approach was AI-suggested, but I wrote the implementation.
- Compares the iterative GD solution against the closed-form least-squares solution `w = (IᵀI)⁻¹ Iᵀy` as a correctness check.
- The feature map `I` is built explicitly from the basis functions applied to each input.

### [neuralnetwork.py](neuralnetwork.py) — Multilayer Perceptron from Scratch

Mostly written by me. A 3-hidden-layer feedforward neural network with sigmoid activations and MSE loss, sized for MNIST digit classification (784 → 16 → 16 → 10). Forward pass and backpropagation are hand-derived and implemented with NumPy.

AI assisted with `calculate_error_signal` and the delta/weight-gradient math. Everything else — structure, forward pass, training loop, test scaffolding — is mine.

What's implemented:

- Forward pass with interleaved pre-activation (`z`) and activation (`σ(z)`) computations.
- Output-layer error signal `δ^L = (a^L - y) ⊙ σ'(z^L)`.
- Backprop through hidden layers via `δ^l = ((W^(l+1))ᵀ δ^(l+1)) ⊙ σ'(z^l)`.
- Weight gradients as outer products `∂C/∂W^l = δ^l ⊗ (a^(l-1))ᵀ`.
- Bias gradients `∂C/∂b^l = δ^l`.

Current state: overfits a single random input to a one-hot target as a correctness check. Real MNIST data loading and minibatch training are not yet wired up.

## Running

```bash
pip install numpy
python regression.py
python neuralnetwork.py
```

## Roadmap

Things I'm planning to add or explore:

- **Finish MNIST training** — mini-batch SGD, real data loading, cross-entropy + softmax output. Details in [future_nn_optimizations.md](future_nn_optimizations.md).
- **Hyperparameter tuning** — experiment with learning rate schedules, optimizer variants (momentum, Adam), and regularization.
- **MCP / agentic development** — higher-level projects using the Model Context Protocol, separate from the from-scratch math work.

## Notes

The from-scratch components prioritize math visibility over performance. Code uses explicit NumPy operations rather than abstracted layer/optimizer classes so the derivations stay readable. More high-level projects (MCP, agentic tools) will use standard libraries as appropriate.
