# Configuration

The primary fields are:

- `method`: `standard` or `spatial`
- `data`: burst location, normalization, batch size, and coordinate scaling
- `model.image`: SIREN image model settings
- `model.psf`: fixed or spatial PSF settings
- `pretraining`: optional image-model warm start before blind deconvolution
- `training`: Lightning trainer and learning-rate settings
- `outputs`: validation arrays and compact diagnostic figures

## Data windows

Loaders return the common `(height, width, frames, channels)` representation. The number
of image channels is inferred from the loaded data, so users do not need to set it in
`model.image`. Use these
fields to keep training focused on a smaller working set:

```yaml
data:
  type: numpy
  path: examples/data/synthetic_burst.npy
  frame_indices: [0, 1, 2, 3]
  channels: [0, 1]
  subframe:
    x: 128
    y: 128
    size: 128
```

`subframe` also accepts `x_range: [start, stop]` and `y_range: [start, stop]`.

## PSF channels

The default PSF is shared across all image channels:

```yaml
model:
  psf:
    representation: parameters
    channel_mode: shared
```

Use `per_channel` when each image channel should learn an independent PSF per frame:

```yaml
model:
  psf:
    channel_mode: per_channel
```

## PSF representation

Standard NeuralBD supports either a direct learnable PSF parameter grid or a SIREN PSF
field:

```yaml
model:
  psf:
    representation: parameters
```

```yaml
model:
  psf:
    representation: siren
    permute_samples: true
```

For the standard method, the SIREN represents `PSF(px, py)`. For the spatial method, the
SIREN represents `PSF(x, y, px, py)`. Spatial NeuralBD therefore always uses
`representation: siren`.

## Progressive training

Progressive training starts with a small PSF support and many training points per batch,
then grows the PSF while reducing the point count and learning rate:

```yaml
training:
  progressive:
    enabled: true
    start_psf_size: 3
    target_psf_size: 65
    n_stages: 6
    training_points_start: 8192
    training_points_end: 1024
    learning_rate_start: 3.0e-4
    learning_rate_end: 3.0e-5
```

The generated stages use odd PSF sizes between `start_psf_size` and `target_psf_size`.
For full control, define explicit stages:

```yaml
training:
  progressive:
    enabled: true
    stages:
      - epochs: 200
        psf_size: 3
        training_points: 8192
        learning_rate: 3.0e-4
      - epochs: 400
        psf_size: 17
        training_points: 4096
        learning_rate: 1.0e-4
      - epochs: 800
        psf_size: 65
        training_points: 1024
        learning_rate: 3.0e-5
```

## Pretraining

Pretraining fits the image SIREN to one reference target before the full NeuralBD
optimization starts. This gives the latent reconstruction a stable initialization while
keeping the PSF untouched.

```yaml
pretraining:
  enabled: true
  epochs: 500
  learning_rate: 1.0e-4
  target: first_frame
  frame_index: 0
```

`target` can be `first_frame`, `frame`, or `mean`.

## Validation figures

Validation writes compact samples only, defaulting to five examples:

```yaml
outputs:
  save_validation_arrays: true
  save_validation_figures: true
  sample_count: 5
  reference_path: null
  reference_array_key: null
```

Figures include learned PSFs, input versus latent reconstruction, convolved prediction
versus observed reference, and optional reconstruction versus ground truth.
