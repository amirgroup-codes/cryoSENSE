# CryoSENSE - Developer Guide

## Project Overview

**CryoSENSE** is a guided diffusion model for accurate cryo-electron microscopy (cryo-EM) super-resolution. It solves the inverse problem of reconstructing biologically accurate high-resolution images from compressed, low-resolution linear measurements.

The system couples an unconditional denoising diffusion probabilistic model (DDPM) trained on cryo-EM data with Nesterov-accelerated gradients to steer the reverse diffusion toward solutions consistent with compressed measurements. CryoSENSE enables high-resolution recovery from inputs up to 32x lower resolution while preserving critical structural information for downstream analysis.

**Key Technologies:**
- Python 3.13.2
- PyTorch 2.7.0
- Diffusers 0.33.1 (Hugging Face)
- CUDA-enabled GPU support

---

## Common Commands

### Environment Setup

**IMPORTANT**: Before running any commands, activate the Conda environment:

```bash
source /usr/scratch/danial_stuff/CondaENVSep6/conda/bin/activate && conda activate /usr/scratch/danial_stuff/anaconda3/envs/cryogen_nov && [YOUR CODE]
```

Example:
```bash
# Run a Python script
source /usr/scratch/danial_stuff/CondaENVSep6/conda/bin/activate && conda activate /usr/scratch/danial_stuff/anaconda3/envs/cryogen_nov && python scripts/run_experiments.py

# Run the CLI
source /usr/scratch/danial_stuff/CondaENVSep6/conda/bin/activate && conda activate /usr/scratch/danial_stuff/anaconda3/envs/cryogen_nov && cryosense --model anon202628/empiar10076-ddpm-ema-cryoem-128x128 --cryoem_path data/sample_empiar10076.pt --block_size 16 --num_masks 50 --start_id 0 --end_id 10 --use_config
```

### Installation

```bash
# From source (recommended for development)
git clone <repository-url>
cd CryoSENSE
pip install -e .

# From PyPI (when available)
pip install cryosense
```

### Basic Usage

```bash
# Quick reconstruction with CLI
cryosense --model anon202628/empiar10076-ddpm-ema-cryoem-128x128 \
        --cryoem_path /path/to/data.pt \
        --block_size 16 \
        --num_masks 50 \
        --start_id 0 \
        --end_id 10 \
        --use_config \
        --verbose

# Run example scripts
python examples/simple_example.py
bash examples/simple_example.sh
```

### Experiment Workflows

```bash
# Run full experiments across multiple parameters
python scripts/run_experiments.py

# Batch reconstruction across entire dataset
python scripts/reconstruct_all_images.py
```

### Training Custom Models

```bash
# Train DDPM model on custom cryo-EM data
cd training
accelerate launch --mixed_precision="bf16" \
  --gpu_ids="0,1,2,3,4,5,6,7" \
  --num_processes=8 \
  --multi_gpu \
  train_unconditional.py \
  --custom_dataset_path="/path/to/particles.mrcs" \
  --output_dir="ddpm-custom-model" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --resolution=128 \
  --use_ema
```

### Comparative Methods

```bash
# Run baseline comparison experiments (DCT, Wavelet, TV, DMPlug)
cd comparative_methods
python run_experiments.py

# Reconstruct full dataset with best parameters
python reconstruct_all_images.py
```

### Analysis & Visualization

```bash
# Generate metrics plots
cd experimental_results
python metrics_plots.py

# CryoDRGN analysis
python cryodrgn_umap_plots.py
python cryodrgn_volumes.py

# ModelAngelo protein structure analysis
python modelangelo_analysis.py
```

### Testing & Development

```bash
# No formal test suite, but you can run the examples to validate:
python examples/simple_example.py

# Check if imports work
python -c "from CryoSENSE import CryoSENSE; print('Success')"
```

---

## High-Level Architecture

### Core Components

#### 1. **Main API (`CryoSENSE/main.py`)**

The `CryoSENSE` class provides the high-level interface:

```python
from CryoSENSE import CryoSENSE

cryosense = CryoSENSE(
    model_path="path/to/ddpm",
    block_size=16,
    device="cuda",
    use_config=True,
    verbose=True
)

reconstructed, original, metrics = cryosense.reconstruct_from_cryoem(
    file_path="data.pt",
    image_ids=[0, 1, 2],
    num_masks=30
)
```

**Key methods:**
- `generate_masks()` - Create binary/Fourier masks for measurements
- `get_measurements()` - Apply measurement operator
- `reconstruct_image()` - Core reconstruction with CryoSENSE algorithm
- `evaluate_reconstruction()` - Calculate PSNR, SSIM, LPIPS metrics
- `reconstruct_from_cryoem()` - End-to-end pipeline

#### 2. **Core Algorithm (`CryoSENSE/core.py`)**

Implements the fundamental CryoSENSE sampling algorithm:

- **`measurement_operator()`** - Forward measurement model
  - Supports time-domain convolution (binary masks)
  - Supports Fourier domain subsampling (complex masks)
  - Uses block-wise sum pooling for downsampling

- **`cryogen_sampling()`** - Main reconstruction algorithm
  - DDPM reverse diffusion process
  - Nesterov momentum for gradient guidance
  - Linear scheduling for `beta` (momentum) and `zeta` (step size)
  - Measurement consistency gradients at each timestep

- **`measurement_consistency_gradient()`** - Computes ∇||y - A(x₀)||²
  - Ensures reconstruction matches compressed measurements

**Algorithm Flow:**
```
1. Initialize x_T ~ N(0, I)
2. For t = T to 1:
   a. Predict noise with UNet: ε = UNet(x_t, t)
   b. Get DDPM estimate: x₀ = scheduler.step(ε, t, x_t)
   c. Compute Nesterov lookahead: x₀' = x₀ - β*momentum
   d. Compute gradient: ∇ = ∂||y - A(x₀')||²/∂x₀'
   e. Update momentum: momentum = β*momentum + ζ*∇
   f. Apply guidance: x_{t-1} = x_prev_ddpm - momentum
3. Return x₀
```

#### 3. **Data Loading (`CryoSENSE/data.py`)**

- **Formats:** PyTorch tensors (`.pt`), MRC files (`.mrcs`)
- **Normalization:** Automatic conversion to [-1, 1] range
- **Batch loading:** Efficient multi-image loading with GPU transfer
- **Noise injection:** Gaussian noise for robustness testing

#### 4. **Mask Generation (`CryoSENSE/masks.py`)**

Supports multiple mask types:
- `random_binary` - Random binary masks (time-domain)
- `random_gaussian` - Gaussian random matrices
- `checkerboard` - Block-wise binary patterns
- `random_fourier` - Random Fourier coefficient sampling
- `fourier_ring` - Weighted ring sampling (biased toward low frequencies)
- `fourier_radial` - Radial spoke patterns

#### 5. **Evaluation (`CryoSENSE/evaluation.py`)**

Metrics computed:
- **MSE/MAE** - Pixel-level error
- **PSNR** - Peak Signal-to-Noise Ratio
- **SSIM** - Structural Similarity Index
- **LPIPS** - Perceptual similarity (Alex network)
- **L2 Norm** - Distance between images
- **Measurement MSE** - Consistency with measurements

Visualization outputs:
- Original vs. Reconstructed comparison
- Error maps with color-coded heatmaps
- Diffusion process GIFs (verbose mode)
- Individual measurement visualizations

#### 6. **Configuration System (`CryoSENSE/config.py`)**

Block-size-specific optimal hyperparameters:

| Block Size | zeta_scale | zeta_min | beta | beta_min |
|------------|------------|----------|------|----------|
| 2, 4, 8, 16 | 1.0 | 1e-2 | 0.9 | 0.1 |
| 32, 64 | 10.0 | 1e-2 | 0.9 | 0.1 |

Config files in `configs/`:
- `default.json` - Base configuration
- `block_size_{2,4,8,16,32}.json` - Size-specific configs

#### 7. **Command-Line Interface (`CryoSENSE/cli.py`)**

Entry point: `cryosense` command (registered in `setup.py`)
- Parses arguments
- Handles batch processing
- Manages GPU memory between batches
- Saves results to CSV and image files

### Directory Structure

```
CryoSENSE/
├── CryoSENSE/                 # Main package
│   ├── __init__.py          # Package exports
│   ├── main.py              # High-level API (CryoSENSE class)
│   ├── core.py              # Core algorithm (sampling, gradients)
│   ├── cli.py               # Command-line interface
│   ├── data.py              # Data loading utilities
│   ├── masks.py             # Mask generation
│   ├── evaluation.py        # Metrics and visualization
│   └── config.py            # Configuration management
│
├── configs/                 # Hyperparameter configs
│   ├── default.json
│   ├── block_size_2.json
│   ├── block_size_4.json
│   ├── block_size_8.json
│   ├── block_size_16.json
│   └── block_size_32.json
│
├── scripts/                 # Experiment scripts
│   ├── run_experiments.py   # Multi-parameter grid search
│   └── reconstruct_all_images.py  # Batch processing
│
├── examples/                # Quick-start examples
│   ├── simple_example.py    # Python API demo
│   └── simple_example.sh    # CLI demo
│
├── training/                # DDPM training
│   ├── train_unconditional.py  # Training script
│   └── train_unconditional.sh  # Training launcher
│
├── comparative_methods/     # Baseline comparisons
│   ├── baselines.py         # DCT, Wavelet, TV, DMPlug
│   ├── comparative_methods_utils.py
│   ├── run_experiments.py
│   ├── reconstruct_all_images.py
│   └── proxTV/              # Total Variation implementation
│
├── experimental_results/    # Analysis & plotting
│   ├── metrics_plots.py     # PSNR, SSIM, LPIPS plots
│   ├── cryodrgn_umap_plots.py  # Latent space analysis
│   ├── cryodrgn_volumes.py  # 3D volume generation
│   ├── modelangelo_analysis.py  # Protein structure analysis
│   ├── cryodrgn_data/       # CryoDRGN model data
│   └── modelangelo_plots_data/  # Protein model files
│
├── data/                    # Sample data
│   ├── sample_empiar10076.pt
│   └── sample_empiar10076.png
│
├── results/                 # Output directory
│   ├── block2_4masks/
│   └── block32_1024masks/
│
├── setup.py                 # Package installation
├── requirements.txt         # Dependencies
└── README.md                # User documentation
```

### Data Flow

```
Input: Compressed Measurements (y) + Binary Masks (M)
           ↓
    [Measurement Operator A]
           ↓
    Target: y = A(x₀) + noise
           ↓
═══════════════════════════════════════════════════════════
    [DDPM UNet + Scheduler]
           ↓
    Initialize: x_T ~ N(0, I)
           ↓
    For each timestep t:
      1. UNet predicts noise ε_θ(x_t, t)
      2. Scheduler estimates x₀
      3. Compute Nesterov lookahead
      4. Calculate measurement gradient ∇
      5. Update with momentum guidance
           ↓
═══════════════════════════════════════════════════════════
    Output: x₀ (Reconstructed Image)
           ↓
    [Evaluation Module]
           ↓
    Metrics: PSNR, SSIM, LPIPS, MSE, MAE
           ↓
    Visualizations: Comparisons, Error Maps, GIFs
```

### Key Design Patterns

1. **Configuration-Driven Parameters**: Block size automatically selects optimal hyperparameters

2. **Modular Measurement Operators**: Supports both time-domain (binary masks) and frequency-domain (Fourier masks) measurements

3. **Flexible Pipeline**: Can be used via Python API or CLI

4. **Batch Processing**: Efficient GPU utilization with batched operations

5. **Verbose Mode**: Optional detailed visualization for debugging and analysis

6. **Lazy Model Loading**: Models can be loaded from HuggingFace Hub on-demand

---

## Important Information

### Pre-trained Models (HuggingFace)

Available DDPM models:
- `anon202628/empiar10076-ddpm-ema-cryoem-128x128` (128×128)
- `anon202628/empiar11526-ddpm-ema-cryoem-128x128` (128×128)
- `anon202628/empiar10166-ddpm-ema-cryoem-128x128` (128×128)
- `anon202628/empiar10786-ddpm-ema-cryoem-128x128` (128×128)
- `anon202628/empiar10648-ddpm-cryoem-256x256` (256×256)

### Datasets (HuggingFace)

- `anon202628/3D_Volumes_EMPIAR10076`
- `anon202628/3D_Volumes_EMPIAR10648`
- `anon202628/EMPIAR10076_128x128`
- `anon202628/EMPIAR11526_128x128`
- `anon202628/EMPIAR10166_128x128`
- `anon202628/EMPIAR10786_128x128`
- `anon202628/EMPIAR10648_256x256`

### Recommended Parameters by Compression Level

| Compression | Block Size | Num Masks | Expected Quality |
|-------------|------------|-----------|------------------|
| Very Low (2×) | 2 | 4-10 | Excellent |
| Low (4×) | 4 | 20-50 | Very Good |
| Medium (8×) | 8 | 50-100 | Good |
| High (16×) | 16 | 100-500 | Moderate |
| Very High (32×) | 32 | 500-1024 | Challenging |

### Parameter Tuning Guidelines

**zeta_scale**: Controls measurement consistency gradient step size
- Larger block sizes need larger values (e.g., 10.0 for block_size=32)
- Smaller block sizes use 1.0
- Too high: unstable, artifacts
- Too low: poor measurement consistency

**beta**: Momentum factor for Nesterov acceleration
- Typically 0.9 (final value)
- Linearly increases from beta_min (0.1) to beta (0.9)
- Higher values = more momentum = faster convergence but less stable

**num_masks**: Number of measurements
- More masks = better reconstruction but slower
- Scale with block size (larger blocks need more masks)
- Minimum: ~1-2 masks per downsampling factor

**num_timesteps**: DDPM sampling steps
- Default: 1000 (full DDPM schedule)
- Can reduce for faster inference (500-100)
- Trade-off: speed vs. quality

### Performance Considerations

1. **GPU Memory**: 
   - 128×128 images: ~4-8GB VRAM per image
   - 256×256 images: ~16-32GB VRAM per image
   - Use batch_size=1 for large images

2. **Computation Time**:
   - Per image: ~2-5 minutes (1000 timesteps, block_size=32)
   - Scales with: num_timesteps, num_masks, block_size

3. **Parallel Processing**:
   - Use `scripts/reconstruct_all_images.py` for multi-GPU
   - Distributes work across available GPUs automatically

### Common Pitfalls

1. **Image Range**: Ensure inputs are in [-1, 1] range (handled automatically)

2. **Block Size Mismatch**: Image size must be divisible by block_size

3. **Mask Type**: Use `random_binary` for time-domain, `random_fourier` for Fourier-domain

4. **Config Usage**: Always use `--use_config` or `use_config=True` for optimal parameters

5. **Verbose Mode Memory**: Saving diffusion steps consumes extra memory/disk

### Extending the System

**Adding New Mask Types:**
1. Edit `CryoSENSE/masks.py`
2. Add new condition in `create_binary_masks()`
3. Return tensor of shape `[num_masks, img_size, img_size]`
4. Use `torch.complex64` for Fourier masks

**Custom Measurement Operators:**
1. Modify `measurement_operator()` in `CryoSENSE/core.py`
2. Ensure differentiability for gradient computation
3. Update `measurement_consistency_gradient()` accordingly

**New Evaluation Metrics:**
1. Add metric computation in `CryoSENSE/evaluation.py`
2. Update `analyze_reconstruction()` function
3. Add to CSV fieldnames in `save_metrics_to_csv()`

### Troubleshooting

**Issue: CUDA out of memory**
- Solution: Reduce batch_size, use smaller images, or fewer masks

**Issue: Poor reconstruction quality**
- Solution: Increase num_masks, check block_size config, verify data normalization

**Issue: Slow convergence**
- Solution: Increase zeta_scale, adjust beta parameters

**Issue: Model not found**
- Solution: Check HuggingFace model name, ensure internet connection

---

## Development Workflow

### Typical Development Tasks

1. **Testing new parameters:**
   ```python
   from CryoSENSE import CryoSENSE
   
   cryosense = CryoSENSE(
       model_path="anon202628/empiar10076-ddpm-ema-cryoem-128x128",
       block_size=16,
       use_config=False  # Manually specify params
   )
   
   results = cryosense.reconstruct_from_cryoem(
       file_path="data/sample.pt",
       image_ids=[0],
       zeta_scale=5.0,  # Custom value
       beta=0.95
   )
   ```

2. **Adding new functionality:**
   - Core algorithm changes: Edit `CryoSENSE/core.py`
   - New masks: Edit `CryoSENSE/masks.py`
   - New metrics: Edit `CryoSENSE/evaluation.py`
   - CLI options: Edit `CryoSENSE/cli.py`

3. **Running experiments:**
   ```bash
   # Edit scripts/run_experiments.py to customize parameters
   python scripts/run_experiments.py
   
   # Results saved to grid search directories
   # Best parameters identified automatically
   ```

4. **Training new models:**
   ```bash
   cd training
   # Edit train_unconditional.sh with your dataset path
   bash train_unconditional.sh
   ```

### Git Workflow

Current branch: `main`

**Commit Guidelines:**
- Core algorithm changes should include validation tests
- Update README.md if CLI options change
- Document new hyperparameters in config files

---

## Additional Resources

- **Paper**: CryoSENSE research paper (NeurIPS 2025 submission)
- **Related Work**: DDPM, Nesterov momentum, cryo-EM reconstruction
- **Dependencies**: See `requirements.txt` for full list

### Key Papers/Concepts

1. **DDPM (Denoising Diffusion Probabilistic Models)**: Foundation for the generative model
2. **Nesterov Momentum**: Accelerated gradient method for measurement consistency
3. **Cryo-EM**: Structural biology imaging technique
4. **CryoDRGN**: Conformational heterogeneity analysis
5. **ModelAngelo**: Atomic model building from cryo-EM maps

---

## Quick Reference

**Python API:**
```python
from CryoSENSE import CryoSENSE

cryosense = CryoSENSE(model_path, block_size, use_config=True, verbose=True)
reconstructed, original, metrics = cryosense.reconstruct_from_cryoem(
    file_path, image_ids, num_masks, num_timesteps
)
```

**CLI:**
```bash
cryosense --model MODEL --cryoem_path DATA --block_size N \
        --num_masks M --start_id 0 --end_id 10 --use_config --verbose
```

**Entry Point:** `CryoSENSE.cli:main` → registered as `cryosense` command

**Core Algorithm:** `CryoSENSE.core:cryogen_sampling`

**Measurement Operator:** `CryoSENSE.core:measurement_operator`

---

*This documentation is intended for developers working on or extending the CryoSENSE codebase. For end-user documentation, see README.md.*
