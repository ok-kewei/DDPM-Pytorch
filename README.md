# Denoising Diffusion Probabilistic Model (DDPM)

## Overview

This project provides a PyTorch implementation of a **Denoising Diffusion Probabilistic Model (DDPM)** with support for both **linear** and **cosine** noise schedules. The primary objective is to investigate how different noise scheduling strategies influence model training dynamics and the quality of generated samples.

**Noise Schedules:**

* **Linear:** Noise increases linearly across timesteps.
* **Cosine:** Noise follows a cosine function, enabling smoother denoising transitions.

All components—including training, sampling, and evaluation—are implemented from scratch for educational and experimental purposes.

---

## Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd diffusion-model
```

2. **Create and activate a Conda environment:**

```bash
conda create -n ddpm python=3.10
conda activate ddpm
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configuration:**
   Edit the configuration file `config/default.yaml` to select the desired noise schedule:

```yaml
train_params:
  noise_scheduler: 'linear'  # Options: 'linear' or 'cosine'
```

---

## Usage

### Training

```bash
python tools/train_ddpm_cifar.py
```

### Sampling

```bash
python tools/sample_ddpm.py
```

### Comparing Noise Schedules

To visualize and compare the effect of linear vs. cosine schedules:

```bash
python image_compare.py
```

The resulting comparison image will be saved as `diffusion_comparison.png`.
