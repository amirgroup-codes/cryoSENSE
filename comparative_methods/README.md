# CryoSENSE Reconstruction: Comparative Methods

This folder provides the codes for running comparative method reconstruction experiments for CryoSENSE project. We perform:

- **Systematic hyperparameter grid search and reconstruction evaluation** (`run_experiments.py`)
- **Full-dataset reconstruction using parallel GPU workers** (`reconstruct_all_images.py`)

All four reconstruction methods, including DCT, Wavelet, TV, and DMPlug, are implemented in `comparative_methods_utils.py`. Overall implementation and analysis is handled by the `baselines.py` script.

---

## Scripts

### `run_experiments.py`

Performs the following:

- Selects a random validation subset of images (consistent with CryoSENSE)
- Runs grid search over `lambda` and `learning_rate` for non-DMPlug baselines
- Saves the best parameters for reconstruction
- Runs final reconstructions on selected validation images

Each run will output:
- `grid_search/{protein}/{comparative_method}/{expt}/lambda_X_lr_Y/`: intermediate reconstruction metrics during hyperparameter search
- `results/{protein}/{comparative_method}/{expt}/`: final reconstructions with the best configuration
- `best_params.json`: stores the best-performing `lambda` and `learning_rate` values for each method

---

### `reconstruct_all_images.py`

Performs the following:

- Loads `best_params.json` from previous experiments
- Uses multiple GPUs (if available) for parallel reconstruction across the entire validation dataset

Each chunk:
- Saves raw reconstructions as `.pt` files
- Also saves `.png` visualizations of each image

---

## Description of folders

### `grid_search/`

Contains grid search results for each protein, baseline method, and mask configuration. An example structure has been left in this folder.

```
grid_search/
└── EMPIAR10786_128/
    └── dct/
        └── block_2_masks_1_random_binary/
            ├── lambda_0.1_lr_0.5/
            │ ├── reconstruction_metrics.csv
            │ └── experiment_log.txt
            ├── lambda_0.1_lr_1.0/
            │ ├── ...
            └── ...
```

### `results/` and `results_{noise_level}/`

Contains final reconstructions using the best hyperparameters (from grid search). An example structure has been left in this folder.

```
results/
└── EMPIAR10786_128/
    └── dct/
        └── block_2_masks_1_random_binary/
            ├── reconstruction_raw_image_0.pt
            ├── reconstruction_raw_image_1.pt
            ├── ...
            └── reconstruction_raw_image_15.pt
```

### `results_3d/`

Contains reconstructions for each protein using the best hyperparameters (from grid search) for each mask configuration. An example structure over 10 images has been left in this folder.

```
results_3d/
└── EMPIAR10786_128/
    └── dct/
        └── block_2_masks_1_random_binary/
            ├── reconstruction_raw_image_0.pt
            ├── reconstruction_raw_image_1.pt
            ├── ...
            └── reconstruction_raw_image_9.pt
```

### `proxTV`

Contains the fast TV proximal gradient implementation from the [proxTV](https://github.com/albarji/proxTV) repository. To implement TV, clone the `proxTV` GitHub into this folder.