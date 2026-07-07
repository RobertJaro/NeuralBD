# NeuralBD

![NeuralBD logo](docs/_static/neuralbd_logo.png)

NeuralBD is a compact framework for neural blind deconvolution of solar image bursts. It learns a sharp latent image and point spread functions directly from a burst of degraded short-exposure frames.

The public core supports two methods:

- **standard NeuralBD**: one learned PSF per burst frame
- **spatial NeuralBD**: coordinate-conditioned, spatially varying PSFs

The implementation is intentionally small: SIREN image model, fixed/spatial PSF models, differentiable convolution, clean data interfaces, Lightning training, and documented extension points.

## Install

```bash
pip install -e ".[dev,docs]"
```

## Quick Start

```bash
python examples/create_synthetic_burst.py
nbd-train --config examples/configs/standard_numpy.yaml
nbd-reconstruct --checkpoint runs/standard_numpy/neuralbd.nbd --out reconstruction.npy
```

## Development Checks

```bash
conda run -n nf2 python -m pytest
LC_ALL=C LANG=C MPLCONFIGDIR=/tmp/matplotlib conda run -n nf2 sphinx-build -b html docs docs/_build/html
```
