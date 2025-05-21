# NeurIPS 2025 Submission - CryoGEN Analysis Scripts

This repository contains scripts and data for analyzing and visualizing the experimental results, as part of the supplementary information for our NeurIPS 2025 paper.

## Contents

The repository contains four main analysis scripts:

1. **cryodrgn_volumes.py** - Generates 3D volumes from cryoDRGN latent spaces for specific particles (Figures 3c, f, i)
2. **cryodrgn_umap_plots.py** - Creates UMAP visualizations with GMM clustering (Figures 3b, e, h)
3. **metrics_plots.py** - Generates LPIPS, PSNR, SSIM Plots for comparing images(Figures 2a, S1, S2, S3)
4. **modelangelo_analysis.py** - Analyzes and compares protein models built with ModelAngelo (Figures 4c-i)

## Directory Structure

```
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── cryodrgn_volumes.py            # Volume generation script
├── cryodrgn_umap_plots.py         # UMAP visualization script
├── metrics_plots.py               # metrics plots visualization script
├── modelangelo_analysis.py        # ModelAngelo analysis script
├── reconstruction_metrics.csv     # Metrics data for plots
├── cryodrgn_data/                 # Input data for cryoDRGN scripts
│   ├── cryogen_z.99.pkl           # Z-values for cryogen dataset
│   ├── original_z.99.pkl          # Z-values for original dataset
│   ├── lowres_z.99.pkl            # Z-values for lowres dataset
│   └── ...                        # Additional data files
├── cryodrgn_volumes/              # Output directory for volumes
├── cryodrgn_plots/                # Output directory for UMAP plots
├── metrics_plots/                 # Output directory for metrics plots
├── modelangelo_plots_data/        # Input data for ModelAngelo analysis
│   ├── modelangelo_output_original.cif
│   ├── modelangelo_output_cryogen.cif
│   ├── modelangelo_output_lowres.cif
│   └── ...
└── modelangelo_plots_outputs/     # Output directory for ModelAngelo plots
```

## Setup and Requirements

### Environment Setup

1. Create a new conda environment:

```bash
conda create -n cryogen_results python=3.9
conda activate cryogen_results
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Install cryoDRGN 3.4.0 (required for volume generation script: cryodrgn_volumes.py):


### Data Download

CryoDRGN model weights are not included directly in this repository due to their size. You need to download these files separately:

1. Download the required weight files from our [HuggingFace repository](https://huggingface.co/anonymousneurips008/CryoDRGN_model_weights/tree/main)

2. Extract the downloaded files to the `cryodrgn_data` directory:


### Requirements

The scripts require the following Python packages (included in requirements.txt):

- numpy>=1.20.0
- matplotlib>=3.5.0
- scikit-learn>=1.0.0
- scipy>=1.7.0
- umap-learn>=0.5.2
- biopython>=1.79
- seaborn>=0.11.0
- pandas>=1.3.0
- torch>=1.9.0
- pickle5
- tqdm>=4.62.0


## Running the Scripts

### 1. CryoDRGN Volume Generation

Generate 3D volumes for specific particles from the CryoGEN, Original, and Low-res datasets (Figures 3c, f, i). Saves the resulting volumes to `cryodrgn_volumes/`:

```bash
python cryodrgn_volumes.py
```

**Note:** This script requires cryoDRGN 3.4.0, and assumes CUDA is available.

### 2. UMAP Visualization

Generate UMAP visualizations with GMM clustering for CryoGEN, Low-res, and Original datasets (Figures 3b, e, h). Saves plots to `cryodrgn_plots/`. Saves correlation analysis between UMAP clusters to `cryodrgn_volumes/cluster_correlations.txt`:

```bash
python cryodrgn_umap_plots.py
```

### 3. Metrics Plots Visualization

Generate plots for LPIPS, PSNR, and SSIM metrics to compare images (Figures 2a, S1, S2, S3). Saves resulting plots to `metrics_plots/`:

```bash
python metrics_plots.py
```

### 4. ModelAngelo Analysis

Analyze and compare protein structures from ModelAngelo between CryoGEN, Original, and Low-res datasets (Figures 4c-i). Saves results to `modelangelo_plots_outputs/`:

```bash
python modelangelo_analysis.py
```

## Expected Outputs

After running the scripts, you should see:

1. **cryodrgn_volumes/** - Contains 3D volumes for selected particles
2. **cryodrgn_plots/** - Contains UMAP visualization plots
3. **cryodrgn_volumes/cluster_correlations.txt** - UMAP cluster correlations
4. **metrics_plots/** - Contains metrics plots
5. **modelangelo_plots_outputs/** - Contains ModelAngelo analysis results
   - confidence_violin_plot.pdf
   - alignment_vs_identity_scatter.pdf
   - chain_alignments.txt
   - lowres_matches_to_original.txt

## Data Generation Process

This section explains how the data files used in this analysis were generated.

### CryoDRGN Models

For the Figure 3 analysis:
- CryoDRGN models were trained using the `cryodrgn train_vae` command on CryoGEN, Original, and Low-res datasets
- This training produced the config files, and z files found in `cryodrgn_data/` and model weights files in the HuggingFace repository

#### CryoDRGN Training Parameters

Each model was trained with identical hyperparameters using a command similar to:

```bash
cryodrgn train_vae [image_stack] --ctf [ctf_file] --poses [poses_file] --zdim 8 -n 100 \
  --enc-dim 1024 --enc-layers 3 --dec-dim 1024 --dec-layers 3 --uninvert-data \
  -o [output_dir] --multigpu -b 32
```

Key parameters:
- **Input data**: Used particle images with their associated CTF parameters and poses
- **Latent dimension**: 8 (zdim=8)
- **Training**: 100 epochs
- **Network architecture**:
  - Encoder: 3 layers with 1024 hidden units each
  - Decoder: 3 layers with 1024 hidden units each
- **Data preprocessing**: Used uninverted data (--uninvert-data)
- **Computational resources**: Multi-GPU (8 A6000 GPUs) training with batch size set to 32

### Volume Reconstructions and Fourier Shell Correlation (FSC) Plots

For Figure 2b, Figure 4a volumes, and Figure 2c, Figure 4b FSC plots:
- Volumes were generated using the CryoDRGN backprojection method via the `cryodrgn backproject` command
- The backprojection command also generated the data for generating FSC plots
- The volumes in Figure 2b can be found in our [HuggingFace repository]([https://huggingface.co/anonymousneurips008/CryoDRGN_model_weights/tree/main](https://huggingface.co/datasets/anonymousneurips008/3D_Volumes_CryoGEN))

### ModelAngelo Atomic Structure Models

For Figure 4c-i analyses:
- Protein models were built using the `modelangelo build` command
- The CIF files in `modelangelo_plots_data/` directory were generated by running ModelAngelo on:
  - Volumes from the CryoGEN, Original, and Low-res reconstructions
  - The FASTA file from PDB 6TTF (corresponding to EMPIAR-10648) was used as the sequence input
