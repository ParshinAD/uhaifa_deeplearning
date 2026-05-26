# Project: NN Initialization & Data Geometry Experiment

## Research context
This is a thesis pre-experiment. The core research question:
**Does the choice of weight initialization scheme destroy the geometric separability
of class data before any training begins?**

Example: two classes (cats vs dogs) may be linearly separable in input space.
After a forward pass through a randomly initialized network, are they still separable
in the latent space — or has initialization already collapsed / scrambled the structure?

## Goal of this notebook
Produce a clean, reproducible Jupyter notebook that:
1. Loads a simple 2-class dataset
2. Initializes a small MLP with multiple init schemes
3. Passes data through (no training) and extracts embeddings
4. Measures and visualizes class separability at each layer

## Tech stack
- Python 3.10+
- PyTorch (model + init schemes)
- scikit-learn (silhouette score, PCA)
- matplotlib + seaborn (plots)
- numpy

## Code style rules
- Every cell must have a markdown cell above it explaining what it does and why
- No magic numbers — all hyperparams defined in a CONFIG dict at the top
- Functions must have docstrings
- Plots must have titles, axis labels, and legends
- Random seeds must be set globally and locally for full reproducibility
- No training — only forward passes (we are studying init, not learning)

## Naming conventions
- `model_*` prefix for network instances
- `emb_*` prefix for embedding tensors
- `score_*` prefix for separability metrics
- Init scheme names: `random_normal`, `xavier_uniform`, `he_normal`, `orthogonal`, `zeros`

## Output artifacts
All plots saved to `./outputs/` with descriptive filenames.
Notebook should be fully runnable top-to-bottom with no errors.