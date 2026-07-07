# Quickstart

This quickstart uses a synthetic NumPy burst to demonstrate the complete training and
export loop.

## Generate example data

```bash
python examples/create_synthetic_burst.py
```

The generated burst follows the NeuralBD data convention:

```text
height, width, frames, channels
```

## Train standard NeuralBD

Train the standard method on the generated burst:

```bash
nbd-train --config examples/configs/standard_numpy.yaml
```

The run writes checkpoints, validation arrays, and sampled validation figures under the
configured `base_dir`.

## Export the reconstruction

```bash
nbd-reconstruct --checkpoint runs/standard_numpy/neuralbd.nbd --out reconstruction.npy
```

## Train the spatial method

The spatially varying method uses a coordinate-conditioned PSF field:

```bash
nbd-train --config examples/configs/spatial_numpy.yaml
```

Use the spatial method when the degradation varies across the field of view.
