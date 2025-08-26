# Denoising Diffusion Probabilistic Model (DDPM)

## Overview

This project provides a PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** with support for both **linear** and **cosine** noise schedules. The goal is to study how different noise scheduling strategies affect training dynamics and the quality of generated samples.

**Noise Schedules:**

* **Linear:** Noise increases linearly across timesteps.
 
    <img src="https://latex.codecogs.com/png.image?\dpi{110}%20%5Cbeta_t%20%3D%20%5Cbeta_1%20%2B%20%5Cfrac%7Bt-1%7D%7BT-1%7D%20(%5Cbeta_T%20-%20%5Cbeta_1)%2C%20t%3D1%2C2%2C...%2CT" alt="linear beta_t formula">
    
    
    Then we define:
    
    <img src="https://latex.codecogs.com/png.image?\dpi{110}%20%5Calpha_t%20%3D%201%20-%20%5Cbeta_t%2C%20%5Cquad%20%5Cbar%7B%5Calpha%7D_t%20%3D%20%5Cprod_%7Bs%3D1%7D%5E%7Bt%7D%20%5Calpha_s" alt="alpha_t formula">

  
* **Cosine:** Noise follows a cosine function for smoother denoising transitions.
    
   <img src="https://latex.codecogs.com/png.image?\dpi{110}%20%5Cbar%7B%5Calpha%7D_t%20=%20%5Cfrac%7B%5Ccos%5E2%28%5Cdfrac%7B%5Cdfrac%7Bt%7D%7BT%7D%20+%20s%7D%7B1%20+%20s%7D%20%5Ccdot%20%5Cfrac%5Cpi%7B2%7D%29%7D%7B%5Ccos%5E2%28%5Cdfrac%7Bs%7D%7B1%20+%20s%7D%20%5Ccdot%20%5Cfrac%5Cpi%7B2%7D%29%7D%2C%20%20%20t=0,1,...,T;%09%09%09s=0.008" alt="cosine alpha_bar_t nested fractions">


    From ᾱₜ, we compute βₜ:
    
    <img src="https://latex.codecogs.com/png.image?\dpi{110}%20%5Cbeta_t%20%3D%20%5Cmin%281%20-%20%5Cfrac%7B%5Cbar%7B%5Calpha%7D_t%7D%7B%5Cbar%7B%5Calpha%7D_%7Bt-1%7D%7D%2C%200.999%29%2C%20t%3D1%2C2%2C...%2CT" alt="cosine beta_t formula">
    
    
    
    <!-- Forward diffusion process:     
    <img src="https://latex.codecogs.com/png.image?\dpi{110}%20x_t%20%3D%20%5Csqrt%7B%5Cbar%7B%5Calpha%7D_t%7D%20x_0%20%2B%20%5Csqrt%7B1%20-%20%5Cbar%7B%5Calpha%7D_t%7D%20%5Cepsilon%2C%20%5Cepsilon%20%5Csim%20%5Cmathcal%7BN%7D%280%2C%20I%29" alt="forward diffusion formula"> -->


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

  



