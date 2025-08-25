# Denoising Diffusion Probabilistic Model (DDPM)

## Overview

This project provides a PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** with support for both **linear** and **cosine** noise schedules. The goal is to study how different noise scheduling strategies affect training dynamics and the quality of generated samples.

**Noise Schedules:**

* **Linear:** Noise increases linearly across timesteps.
* **Cosine:** Noise follows a cosine function for smoother denoising transitions.

All components—including training, sampling, and evaluation—are implemented from scratch for educational and experimental purposes.

## Installation

```bash
# Clone repository
git clone
cd diffusion-model

# Create and activate Conda environment
conda create -n ddpm python=3.10
conda activate ddpm

# Install dependencies
pip install -r requirements.txt
```

Edit `config/default.yaml` to set the noise schedule and hyperparameters (e.g., learning rate, batch size, number of timesteps):

```yaml
train_params:
  noise_scheduler: 'linear'  # Options: 'linear' or 'cosine'
  learning_rate: 0.0002
  batch_size: 128
  num_timesteps: 1000
```

## Usage

```bash
# Train the model
python tools/train_ddpm_cifar.py

# Generate samples
python tools/sample_ddpm.py

# Compare linear vs. cosine schedules
python image_compare.py
# Output: diffusion_comparison.png
```

## Results
* **Dataset:** The model was trained on **CIFAR-10** at **32×32 resolution**, so the generated image quality is limited and not as sharp as higher-resolution datasets.  
* **Training behavior:** Linear schedules generally show faster convergence, while cosine schedules enable smoother denoising.
* **Sample quality:** Cosine schedules tend to produce sharper, more coherent images compared to linear schedules.
* **Visualization:** The script `image_compare.py` generates side-by-side comparisons of denoising progress.

Example (linear vs. cosine schedule):

<p align="left">
   <img width="600" height="1000" alt="diffusion_comparison" src="https://github.com/user-attachments/assets/4cf560b4-5dcf-4c3b-a177-29041cf9da3f" />
</p>

### Comparison Across Timesteps

**Early timesteps (t=999 -> 900)**  
- **Linear:** Completely random noise; structure is almost totally destroyed until much later (around t=500).  
- **Cosine:** Even at t=950 or 900, faint silhouettes are visible. Cosine applies less noise early on, preserving structure longer.  

**Mid timesteps (t=700 -> 300)**  
- **Linear:** Produces blurry, low-contrast blobs; shapes are hard to distinguish.  
- **Cosine:** Reconstructions are much clearer and retaining semantic information.  

**Late timesteps (t=100 -> 0)**  
- **Linear:** Final images are clearer but often washed out or lower contrast.  
- **Cosine:** Final images are sharper and better defined — exactly what cosine is designed for, yielding sharper convergence at the end

  
