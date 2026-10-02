# Neural Networks From Scratch

A hands-on, from-first-principles walk through how neural networks and language models actually work — no black boxes. Every core mechanism (automatic differentiation, backpropagation, gradient descent, embeddings, softmax) is built by hand in plain Python/NumPy/PyTorch tensors before any high-level shortcuts are used.

This repo is structured like a short book: three chapters, each building on the last. If you're learning deep learning and want to understand *why* it works instead of just calling `.fit()`, start at Chapter 1 and work forward.

## Who this is for

Anyone comfortable with basic Python and a little calculus (derivatives) who wants to actually understand:
- What a gradient is and how backpropagation computes it
- Why neural networks are just "differentiable programs"
- How language models like GPT's ancestors were built, step by step

## Chapter 1 — `neural_net.ipynb`: Micrograd — Autograd and Backpropagation by Hand

**The core question this chapter answers: what *is* backpropagation, really?**

You start with a plain scalar function and numerically estimate its derivative using the limit definition (`h -> 0`), to build intuition for what a gradient even means before any framework is introduced.

From there, the notebook builds a `Value` class — a tiny autograd engine. Each `Value` wraps a number and remembers:
- what operation created it (`+`, `*`, etc.)
- which other `Value`s were its inputs

This turns every calculation into a small computation graph. Once the graph exists, a `.backward()`-style pass can walk it in reverse and compute the gradient of every node with respect to the final output — this *is* backpropagation, and here you build it yourself rather than importing it.

Highlights in this chapter:
- Manual gradient checking against PyTorch tensors (`requires_grad=True`) to verify the hand-built engine is correct
- Visualizing the computation graph with `graphviz`, so you can *see* the chain rule propagate backward
- A `Neuron` class built directly on top of `Value` — the first real "neural network" in the repo, made of nothing but the autograd engine you just wrote

**Why it matters:** every modern deep learning framework (PyTorch, TensorFlow, JAX) is built on exactly this idea at its core, just heavily optimized. Once you've built it yourself, `loss.backward()` stops being magic.

## Chapter 2 — `makemore.ipynb`: A Character-Level Language Model, the Simple Way

**The core question: how do you predict the next character in a name?**

Using a dataset of names (`names (1).txt`), this chapter builds a language model in two stages:

1. **Counting-based bigram model.** Literally count how often each character follows another, turn those counts into probabilities, and sample new "names" from those probabilities. This is the simplest possible language model — no learning, just statistics.
2. **The same model, reframed as a neural network.** The bigram counting table is replaced with a single trainable weight matrix. Characters are one-hot encoded, passed through a linear layer, converted to probabilities with `softmax` (via `exp` and normalization), and evaluated using **negative log-likelihood loss** — the same loss function used to train real language models today.

From there, the notebook trains this tiny network with plain gradient descent: compute the loss, call `.backward()`, and nudge the weights (`w.data += -0.1 * w.grad`).

**Why it matters:** this is the bridge between "counting statistics" and "gradient-based learning." It shows that a neural network trained with gradient descent can *rediscover* the same solution a simple counting model finds by hand — which is a great way to build trust that the training process is actually working.

## Chapter 3 — `makemore2.ipynb`: A Real Multilayer Perceptron Language Model

**The core question: what if the model could use more than one previous character?**

This chapter upgrades the model to follow the architecture from Bengio et al.'s 2003 neural language model paper:

- Characters are mapped into a low-dimensional **embedding** space (`C = torch.randn((27, 2))`) instead of raw one-hot vectors
- A **block size** of 3 previous characters is used as context, so the model predicts each next character from a short window of history
- Embeddings are concatenated and passed through a hidden layer with a `tanh` activation
- A second layer projects to logits over the 27 possible output characters, followed by `softmax`
- The model is trained with **mini-batch gradient descent** (random batches of 32 examples per step) instead of computing gradients over the full dataset each time
- Data is properly split into **train / dev** sets to check the model isn't just memorizing

**Why it matters:** this is structurally the same recipe (embed → hidden layer → nonlinearity → output layer → loss → backprop) used in modern deep learning, just at a much smaller scale. Once you understand this notebook, the leap to larger architectures is mostly "more layers, more data, more compute" rather than genuinely new ideas.

## Suggested order

1. `neural_net.ipynb` — build the engine (autograd + backprop)
2. `makemore.ipynb` — build the simplest possible language model, twice (counting, then learned)
3. `makemore2.ipynb` — build a real MLP language model with embeddings and mini-batching

## Requirements

```
numpy
matplotlib
torch
graphviz          # for the computation-graph visualization in Chapter 1
```

## Data

`names (1).txt` — a newline-separated list of names, used as training data for the character-level models in Chapters 2 and 3.

## `graph/`

Contains the rendered computation-graph visualizations produced in Chapter 1.

---

*This repo is a learning project — code favors clarity over performance, and is meant to be read alongside experimenting in the notebooks, not used as a production library.*
