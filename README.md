<p align="center">
  <img src="NdLinearGated.png" alt="Logo" width="400">
  <br /> <br />
</p>

# NdLinearGated: Adaptive Compute Extension for NdLinear

**Note:** Find a detailed report for the project [here](https://drive.google.com/file/d/1n4ddA71mTe2Y47fdsvnB6mH1G_WJWEwd/view?usp=sharing).

## Overview

This repository extends the original [`NdLinear`](https://github.com/ensemble-core/NdLinear) module from Ensemble AI by adding learnable **per-mode gating** for input-dependent computation. 

While NdLinear factors linear transformations across tensor modes (e.g., spatial, temporal), it still processes all modes for every input. NdLinearGated lets the network decide which modes to compute — dynamically skipping projections that aren’t useful for a given input. This adds structured sparsity for **better speed-accuracy tradeoffs**, without breaking the original API.

If NdLinear is a smarter replacement for `nn.Linear`, NdLinearGated is the next step: **modular, interpretable, and selectively sparse.**

## Key Features

- **Input-Adaptive Execution:** Learns when to apply or skip mode-wise projections
- **Plug-in Gating:** Drop-in extension to existing NdLinear layers
- **Soft and Hard Gating:** Choose between differentiable (soft) or binary (hard) control
- **Top-k Gating Heuristics:** Gate only the most variable tensor modes
- **No Changes to Model Logic:** Gates are trained alongside your main task with no extra supervision

## Why Gating?

NdLinear applies independent projections per mode:
```python
x -> X x W_0 x W_1 ... x W_n
```

But do we always need *all* of them? Probably not.

NdLinearGated lets you learn a gate \( \alpha_i(x) \in [0, 1] \) for each mode:
```math
y_i = \alpha_i(x) \cdot (x \times_i W_i) + (1 - \alpha_i(x)) \cdot x
```
This means that for some inputs, mode-$i$ projections can be skipped (identity passthrough) or softly blended.


## Installation

```bash
git clone https://github.com/your-username/NdLinearGated.git
cd NdLinearGated
pip install -e .
```

## Usage

Use it exactly like NdLinear but with gating behavior:

```python
from ndlinear import NdLinearGated

layer = NdLinearGated(
    input_dims=(28, 28, 3),
    hidden_dims=(64, 64, 6),
    gating_mode="topk",     # options: "all", "first", "topk"
    gating_type="soft",     # options: "soft", "hard"
    gate_mlp_hidden_dim=16
)

output = layer(input_tensor)
```

### Gating Configuration
- `gating_type`: 
  - `soft`: uses sigmoid, allows gradients
  - `hard`: thresholded at 0.5 (non-differentiable)

- `gating_mode`: 
  - `all`: apply gates to all tensor modes
  - `first`: only gate the first mode
  - `topk`: gate only the top-2 highest-variance modes (auto-selected)


## Project Structure

```bash
ndlinear/
 ├─ modules/            # Core NdLinear + Gated variant
 ├─ experiments/        # Training + evaluation logic
 └─ visualization/      # Gate entropy, activity plots
```


## Experiments

We provide ablations on:
- CIFAR10 (vision)
- MNIST (vision)
- AG News (text)
- IMDB (text)

To run the full ablation suite:
```bash
python experiments/run_ablation.py --config experiments/configs/ablation.yml
```

To generate visualizations:
```bash
python visualization/visualize.py --output_dir outputs
```


## Results (Highlights)

| Dataset  | Variant                        | Accuracy | Proxy Compute |
|----------|--------------------------------|----------|---------------|
| CIFAR10  | NdLinearGated (soft, topk, 16) | 42.0%    |  < 50%        |
| IMDB     | NdLinearGated (soft, topk, 64) | 83.88%   |  < 50%        |
| AG News  | All Gated Variants             | >90%     |  25-50%       |

Gating reduced unnecessary computation **without hurting performance**. Soft gating trained faster and generalizes better. Gate entropy logs show confident selection after 3–4 epochs.
Please refer to outputs/ for other ablation plots and statistics.

## Requirements

- Python >= 3.7
- PyTorch >= 1.8
- torchvision
- datasets
- transformers
- matplotlib, seaborn
- numpy, pandas, pyyaml


## License

Apache 2.0. See `LICENSE` for details.


## Credits

NdLinear was originally developed by [Ensemble AI](https://ensemblecore.ai/). NdLinearGated was developed by [Aayush Upadhyay](https://github.com/aaupadhy) and it builds on top of Ensemble's open-source foundation.
