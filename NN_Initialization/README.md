# NN Initialization & Data Geometry Experiment

Does weight initialization destroy class separability before training starts?

## Run

```bash
jupyter nbconvert --to notebook --execute init_geometry_experiment.ipynb --output init_geometry_experiment.ipynb
```

Plots are saved to `./outputs/`. Requires Python 3.10+, PyTorch, scikit-learn, matplotlib, seaborn.

## Install dependencies

```bash
pip install torch scikit-learn matplotlib seaborn pandas jupyter
```

## Outputs

| File | Description |
|---|---|
| `outputs/00_raw_data.png` | PCA 2D scatter of raw input data |
| `outputs/01_silhouette_by_layer.png` | Main result: silhouette score per layer per init scheme |
| `outputs/02_embeddings_grid.png` | Scatter grid: rows = schemes, cols = layers |
| `outputs/03_class_overlap_heatmap.png` | Heatmap summary — thesis-ready figure |
| `outputs/04_zeros_collapse.png` | Norm collapse for zeros init (pathological case) |
