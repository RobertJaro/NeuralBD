# Workflow

NeuralBD reconstructs a latent high-resolution image from a burst of degraded observations.
The workflow is organized around explicit stages: data preparation,
model configuration, optional pretraining, blind deconvolution, validation, and export.

## 1. Prepare the burst

The training pipeline expects a burst in channels-last form:

```text
height, width, frames
```

or:

```text
height, width, frames, channels
```

All images should be strictly positive. The default image model uses a positive
`softplus` output activation, which matches the expected data domain.

Instrument-specific loaders convert supported source formats into this standard layout.
The core training code does not expose configurable axis ordering; loaders are responsible
for returning channels-last arrays.

## 2. Select the working data

Use the `data` section to choose the input path, normalization, frame subset, channel
subset, and optional subframe:

```yaml
data:
  type: numpy
  path: examples/data/synthetic_burst.npy
  normalization: minmax
  frame_indices: [0, 1, 2, 3]
  channels: [0]
  subframe:
    x: 128
    y: 128
    size: 128
```

Subframes are useful for fast experiments, memory control, and progressive development of
new instrument loaders.

## 3. Choose the method

Use the standard method when a burst can be modeled with one PSF per frame:

```yaml
method: standard
```

Use the spatial method when the PSF changes across the field of view:

```yaml
method: spatial
```

The spatial method uses a SIREN PSF field that evaluates `PSF(x, y, px, py)`.

## 4. Configure the PSF

Standard NeuralBD can use direct learnable PSF parameters:

```yaml
model:
  psf:
    representation: parameters
    size: 65
    channel_mode: shared
```

or a continuous SIREN representation:

```yaml
model:
  psf:
    representation: siren
    size: 65
    channel_mode: per_channel
    permute_samples: true
```

`channel_mode: shared` learns one PSF per frame and applies it to all channels.
`channel_mode: per_channel` learns a separate PSF per frame and channel.

## 5. Optional pretraining

Pretraining fits the image SIREN before optimizing the full blind deconvolution model.
This gives the reconstruction a stable initialization and keeps the PSF fixed during the
warm start.

```yaml
pretraining:
  enabled: true
  epochs: 500
  learning_rate: 1.0e-4
  target: first_frame
```

Available targets are `first_frame`, `frame`, and `mean`.

## 6. Progressive training

Progressive training starts with a restricted PSF support and many training points per batch,
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

This schedule is especially useful when the target PSF support is large.

## 7. Validate and export

Validation writes sampled diagnostic figures and optional NumPy arrays. The default
figures include learned PSFs, input versus reconstruction, predicted observations versus
reference observations, and optional comparison with a ground-truth reconstruction.

After training, export the latent image with:

```bash
nbd-reconstruct --checkpoint runs/standard_numpy/neuralbd.nbd --out reconstruction.npy
```

The checkpoint stores model state and reconstruction metadata, not the full input burst.
